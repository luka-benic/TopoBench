"""Callback that saves the test-set predictions of an OpenQDC training run."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from topobench.data.utils.openqdc_utils import method_names_for
from topobench.utils.openqdc_results import (
    log_results_to_wandb,
    save_test_results,
)

log = logging.getLogger(__name__)


class OpenQDCTestResults(Callback):
    """Save per-molecule test predictions and a parity plot when training ends.

    When the fit ends, the best checkpoint is loaded and run once over the test
    set. Three files are written to ``<output_dir>/test_results``:
    ``test_predictions.txt`` (reference and predicted energy of each molecule),
    ``test_metrics.txt`` and ``test_parity.png``. When a Weights & Biases logger
    is used, the parity plot, the metrics and the ``.txt`` files are also
    logged to that run.

    The model is trained on the baseline-subtracted energy ``y``. The
    prediction is converted back to the energy in eV with the stored
    ``energy`` of each molecule, so that the files hold energies and not
    baseline-subtracted values. If the batches carry no ``energy``, the
    baseline-subtracted values are saved instead and the file header says so.

    Parameters
    ----------
    dataset : str
        OpenQDC dataset name, used in the file headers and the plot title.
    model : str
        TopoBench model name, used in the file headers and the plot title.
    energy_method : int
        Energy column of the dataset, used in the file headers.
    output_dir : str, optional
        Folder that receives ``test_results``; default: the trainer's root
        directory.
    per_atom : bool
        True if the target is the energy per atom. The saved energies are then
        multiplied by the number of atoms, so that the files hold energies per
        molecule like those of MACE.
    """

    def __init__(
        self,
        dataset: str = "",
        model: str = "",
        energy_method: int = 0,
        output_dir: str | None = None,
        per_atom: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.energy_method = energy_method
        self.output_dir = output_dir
        self.per_atom = per_atom

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run the best checkpoint over the test set and save the results.

        Parameters
        ----------
        trainer : Trainer
            The Lightning trainer that has just finished fitting.
        pl_module : LightningModule
            The model; the best checkpoint is loaded into it.
        """
        best = getattr(trainer.checkpoint_callback, "best_model_path", "")
        if not best or not os.path.exists(best):
            log.warning("No best checkpoint found: test results not saved.")
            return
        state = torch.load(
            best, map_location=pl_module.device, weights_only=False
        )
        pl_module.load_state_dict(state["state_dict"], strict=True)

        columns = self._predict(
            pl_module, trainer.datamodule.test_dataloader()
        )
        if columns["energy"] is not None:
            reference = columns["energy"]
            prediction = columns["energy"] - (columns["y"] - columns["pred"])
            quantity, note = "energy", ""
        else:
            reference, prediction = columns["y"], columns["pred"]
            quantity = "baseline-subtracted energy"
            note = "original energy not available in the batches"

        if self.per_atom and columns["n_atoms"] is not None:
            reference = reference * columns["n_atoms"]
            prediction = prediction * columns["n_atoms"]
            note = "per-atom target multiplied by the number of atoms"

        out_dir = Path(self.output_dir or trainer.default_root_dir)
        info = {
            "model": self.model,
            "dataset": self.dataset,
            "energy_column": self._column_label(),
            "quantity": quantity,
            "note": note,
        }
        saved = save_test_results(
            out_dir / "test_results",
            reference,
            prediction,
            n_atoms=columns["n_atoms"],
            index=columns["idx"],
            info=info,
        )
        metrics = saved["metrics"]
        message = (
            f"[results] test set: {int(metrics['n'])} molecules, "
            f"MAE {metrics['mae_eV']:.5g} eV; files in {out_dir / 'test_results'}"
        )
        print(message, flush=True)
        log.info(message)
        # With a Weights & Biases logger, the results go to the same run.
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                log_results_to_wandb(logger.experiment, saved)

    def _column_label(self) -> str:
        """Describe the energy column by index and name.

        Returns
        -------
        str
            The index and, if known, the name of the energy column.
        """
        try:
            name = method_names_for(self.dataset)[int(self.energy_method)]
        except (ValueError, IndexError):
            return str(self.energy_method)
        return f"{self.energy_method}: {name}"

    @staticmethod
    def _predict(
        pl_module: LightningModule, loader: Any
    ) -> dict[str, np.ndarray | None]:
        """Run the model over a dataloader and collect predictions.

        Parameters
        ----------
        pl_module : LightningModule
            The model with the best weights loaded.
        loader : Any
            Dataloader over the test set.

        Returns
        -------
        dict[str, numpy.ndarray or None]
            ``pred`` and ``y`` (prediction and target of the model), and
            ``energy``, ``idx`` and ``n_atoms`` (None if the batches do not
            have them).
        """
        pl_module.eval()
        parts: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("pred", "y", "energy", "idx", "n_atoms")
        }
        attributes = {
            "energy": "energy",
            "idx": "openqdc_idx",
            "n_atoms": "n_atoms",
        }
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(pl_module.device)
                pl_module.state_str = "Test"
                out = pl_module.model_step(batch)
                parts["pred"].append(out["logits"].detach().cpu().reshape(-1))
                parts["y"].append(out["labels"].detach().cpu().reshape(-1))
                for key, attribute in attributes.items():
                    value = getattr(batch, attribute, None)
                    if value is not None:
                        parts[key].append(value.detach().cpu().reshape(-1))
        # model_step feeds the metric accumulator of the model; leave it clean
        # for the validation and test of the best checkpoint that follow.
        pl_module.evaluator.reset()

        n = sum(len(t) for t in parts["pred"])
        columns: dict[str, np.ndarray | None] = {}
        for key, tensors in parts.items():
            joined = torch.cat(tensors).numpy() if tensors else None
            if joined is not None and len(joined) != n:
                joined = None  # attribute missing in some batches
            columns[key] = joined
        return columns
