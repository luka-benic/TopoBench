"""Run MACE's training (``mace_run_train``) correctly for energy-only data.

MACE has no energy-only mode on its command line, and two things go wrong for
data without forces. This launcher fixes both before it starts the normal
``mace.cli.run_train.main``; all command-line arguments are passed on
unchanged.

1. With ``--compute_forces=False`` MACE does not compute forces, which saves
   the force gradient (about half of the time of a training step). But its
   energy losses (``weighted``, ``l1l2energyforces``) still read the predicted
   forces and crash with ``TypeError``. The force terms of the loss are
   therefore skipped when no forces were computed.
2. MACE logs the force error of force-less data as a placeholder object
   (``NoneMultiply``) that ``wandb.log`` cannot serialise, so a run with
   ``--wandb`` crashes after the first validation. Such placeholders are turned
   into ``None`` before they reach ``wandb.log``.

The patches target MACE 0.3.x; if the loss functions are not found (another
MACE version), the launcher stops with a clear message.
"""

import functools
from collections.abc import Callable
from typing import Any

import mace.modules.loss as mace_loss
import torch
import wandb
from mace.cli.run_train import main

# Force terms of MACE's energy losses (see mace/modules/loss.py).
FORCE_TERMS = ("mean_squared_error_forces", "mean_normed_error_forces")


def skip_without_forces(function: Callable) -> Callable:
    """Wrap a force-loss function so it returns 0 when forces are not computed.

    Parameters
    ----------
    function : Callable
        A MACE force-loss function ``function(ref, pred, ...)``.

    Returns
    -------
    Callable
        The wrapped function: 0 (scalar, dtype and device of the energy) if
        ``pred["forces"]`` is None, otherwise the result of ``function``.
    """

    @functools.wraps(function)
    def wrapped(ref: Any, pred: Any, *args: Any, **kwargs: Any) -> Any:
        if pred.get("forces") is None:
            energy = ref["energy"]
            return torch.zeros((), dtype=energy.dtype, device=energy.device)
        return function(ref, pred, *args, **kwargs)

    return wrapped


def patch_force_losses() -> None:
    """Make the force terms of MACE's losses skip when forces are not computed."""
    missing = [name for name in FORCE_TERMS if not hasattr(mace_loss, name)]
    if missing:
        raise SystemExit(
            f"run_mace_train.py: {missing} not found in mace.modules.loss; this "
            "MACE version is not supported by the energy-only patch."
        )
    for name in FORCE_TERMS:
        setattr(mace_loss, name, skip_without_forces(getattr(mace_loss, name)))


def clean_metric(value: Any) -> Any:
    """Replace MACE's placeholder for a missing metric by None.

    Parameters
    ----------
    value : Any
        A logged value, possibly a dictionary of values.

    Returns
    -------
    Any
        The value with every ``NoneMultiply`` placeholder replaced by None.
    """
    if isinstance(value, dict):
        return {key: clean_metric(item) for key, item in value.items()}
    return None if type(value).__name__ == "NoneMultiply" else value


_original_init = wandb.init


def _init_with_safe_log(*args: Any, **kwargs: Any) -> Any:
    """Start a run like ``wandb.init`` and make ``wandb.log`` safe."""
    run = _original_init(*args, **kwargs)
    run_log = run.log  # wandb.init has just pointed wandb.log at this

    def safe_log(data: Any, *log_args: Any, **log_kwargs: Any) -> Any:
        return run_log(clean_metric(data), *log_args, **log_kwargs)

    wandb.log = safe_log
    return run


patch_force_losses()
wandb.init = _init_with_safe_log

if __name__ == "__main__":
    main()
