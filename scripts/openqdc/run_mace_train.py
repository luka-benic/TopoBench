"""Run ``mace_run_train`` with Weights & Biases logging that works without forces.

For data without forces MACE logs the force error as a placeholder object
(``NoneMultiply``) that ``wandb.log`` cannot serialise, so a run with
``--wandb`` crashes after the first validation. This launcher turns such
placeholders into ``None`` before they reach ``wandb.log`` and then starts the
normal ``mace_run_train``; all command-line arguments are passed on unchanged.
"""

from typing import Any

import wandb
from mace.cli.run_train import main

_original_init = wandb.init


def _clean(value: Any) -> Any:
    """Replace MACE's placeholder for a missing metric by None."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return None if type(value).__name__ == "NoneMultiply" else value


def _init_with_safe_log(*args: Any, **kwargs: Any) -> Any:
    """Start a run like ``wandb.init`` and make ``wandb.log`` safe."""
    run = _original_init(*args, **kwargs)
    run_log = run.log  # wandb.init has just pointed wandb.log at this

    def safe_log(data: Any, *log_args: Any, **log_kwargs: Any) -> Any:
        return run_log(_clean(data), *log_args, **log_kwargs)

    wandb.log = safe_log
    return run


wandb.init = _init_with_safe_log

if __name__ == "__main__":
    main()
