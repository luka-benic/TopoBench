"""Evaluate a trained MACE model on the test .xyz file and save the results.

``mace_run_train`` only prints a test error table. This module runs the best
model (saved by ``mace_run_train``) over ``test.xyz`` once more and writes
the per-molecule predictions (``test_predictions.txt``), the metrics
(``test_metrics.txt``) and a parity plot (``test_parity.png``) into
``<run folder>/test_results``. It also compares its MAE with the one in
MACE's own test table.
"""

import re
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace import data as mace_data
from mace import tools
from mace.tools import torch_geometric

from topobench.data.utils.openqdc_utils import method_names_for
from topobench.utils.openqdc_results import (
    log_results_to_wandb,
    save_split,
    save_test_results,
)

ENERGY_KEY = "REF_energy"


def predict_test_set(
    model_path: str, xyz_path: str, device: str, batch_size: int = 128
) -> dict[str, np.ndarray]:
    """Predict the energies of all frames of an .xyz file with a MACE model.

    Parameters
    ----------
    model_path : str
        The ``.model`` file written by ``mace_run_train``.
    xyz_path : str
        Extended-XYZ file with ``REF_energy`` and ``openqdc_index``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    batch_size : int
        Number of molecules per batch.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``reference`` and ``prediction`` (eV), ``n_atoms`` and ``index``.
    """
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device).eval()
    dtype = next(model.parameters()).dtype
    tools.set_default_dtype("float64" if dtype == torch.float64 else "float32")

    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)
    frames = read(xyz_path, index=":")
    data = []
    for atoms in frames:
        config = mace_data.Configuration(
            atomic_numbers=np.asarray(atoms.numbers, dtype=int),
            positions=np.asarray(atoms.positions, dtype=float),
            properties={"energy": float(atoms.info[ENERGY_KEY])},
            property_weights={"energy": 1.0, "forces": 0.0},
            pbc=(False, False, False),
        )
        data.append(
            mace_data.AtomicData.from_config(
                config, z_table=z_table, cutoff=r_max
            )
        )
    loader = torch_geometric.dataloader.DataLoader(
        data, batch_size=batch_size, shuffle=False
    )
    predictions = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch.to(device).to_dict(),
                training=False,
                compute_force=False,
            )
            predictions.append(out["energy"].detach().cpu().numpy())
    return {
        "reference": np.array([a.info[ENERGY_KEY] for a in frames]),
        "prediction": np.concatenate(predictions),
        "n_atoms": np.array([len(a) for a in frames]),
        "index": np.array([a.info["openqdc_index"] for a in frames]),
    }


def save_mace_split(
    work_dir: str, name: str, xyz_paths: dict[str, str]
) -> Path:
    """Save the molecules of the train, valid and test files of a MACE run.

    Parameters
    ----------
    work_dir : str
        MACE run folder; ``split.txt`` goes to ``<work_dir>/test_results``.
    name : str
        MACE run name, used in the file header.
    xyz_paths : dict[str, str]
        Paths of the ``train``, ``valid`` and ``test`` ``.xyz`` files.

    Returns
    -------
    Path
        The written file.
    """
    split_ids = {
        split: [
            int(i)
            for i in re.findall(
                r"openqdc_index=(\d+)", Path(xyz_paths[split]).read_text()
            )
        ]
        for split in ("train", "valid", "test")
    }
    return save_split(
        Path(work_dir) / "test_results",
        split_ids,
        ["OpenQDC split of this run", "model: MACE", f"run: {name}"],
    )


def mace_reported_test_mae(log_file: Path) -> tuple[float, bool] | None:
    """Read the test MAE from MACE's own final test table.

    MACE decides which table it prints: the total energy error per molecule
    (``MAE E / meV``) or, for some losses such as ``huber``, the error per
    atom (``MAE E / meV / atom``).

    Parameters
    ----------
    log_file : Path
        The MACE run log.

    Returns
    -------
    tuple[float, bool] or None
        The MAE in meV of the last test table and whether it is per atom, or
        None if there is no MAE table.
    """
    if not log_file.exists():
        return None
    text = log_file.read_text()
    start = text.rfind("Error-table on TEST")
    if start < 0:
        return None
    block = text[start:]
    header = re.search(r"\|\s*config_type\s*\|\s*(MAE E[^|]*)\|", block)
    row = re.search(r"\|\s*\S+\s*\|\s*([-+0-9.eE]+)\s*\|", block)
    if header is None or row is None:
        return None
    return float(row.group(1)), "/ atom" in header.group(1)


def log_mace_results_to_wandb(
    saved: dict,
    run_id: str,
    project: str,
    entity: str,
    directory: str,
    config: dict,
) -> None:
    """Add the test-set results to the Weights & Biases run of a MACE run.

    ``mace_run_train`` has already ended its own run, so the run is resumed by
    its id.

    Parameters
    ----------
    saved : dict
        The result of ``save_test_results``.
    run_id : str
        Id of the Weights & Biases run that ``mace_run_train`` created.
    project : str
        Weights & Biases project.
    entity : str
        Weights & Biases entity (empty for the default).
    directory : str
        Folder for the Weights & Biases files (the one given to MACE).
    config : dict
        Extra settings added to the run's config.
    """
    import wandb

    run = wandb.init(
        project=project or None,
        entity=entity or None,
        id=run_id,
        resume="allow",
        dir=directory,
    )
    run.config.update(config, allow_val_change=True)
    log_results_to_wandb(run, saved)
    run.finish()


def save_mace_test_results(
    work_dir: str,
    name: str,
    test_xyz: str,
    dataset: str,
    energy_method: int,
    device: str,
    batch_size: int,
    seed: int = 0,
) -> dict:
    """Evaluate the best MACE model on the test set and save the results.

    Parameters
    ----------
    work_dir : str
        MACE run folder.
    name : str
        MACE run name (the model is ``<work_dir>/<name>.model``).
    test_xyz : str
        The ``test.xyz`` file of the run.
    dataset : str
        OpenQDC dataset name.
    energy_method : int
        Energy column.
    device : str
        ``"cuda"`` or ``"cpu"``.
    batch_size : int
        Number of molecules per batch.
    seed : int
        Seed of the run; MACE names its log file ``<name>_run-<seed>.log``.

    Returns
    -------
    dict
        Paths of the saved files and the computed metrics.
    """
    model_path = Path(work_dir) / f"{name}.model"
    if not model_path.exists():
        raise FileNotFoundError(f"MACE model not found: {model_path}")
    result = predict_test_set(str(model_path), test_xyz, device, batch_size)
    try:
        column = f"{energy_method}: {method_names_for(dataset)[energy_method]}"
    except IndexError:
        column = str(energy_method)
    saved = save_test_results(
        Path(work_dir) / "test_results",
        result["reference"],
        result["prediction"],
        n_atoms=result["n_atoms"],
        index=result["index"],
        info={
            "model": "MACE",
            "dataset": dataset,
            "energy_column": column,
            "quantity": "energy",
        },
    )
    mae = saved["metrics"]["mae_meV"]
    print(
        f"[results] test set: {int(saved['metrics']['n'])} molecules, MAE "
        f"{mae:.5g} meV; files in {Path(work_dir) / 'test_results'}",
        flush=True,
    )
    reported = mace_reported_test_mae(
        Path(work_dir) / "logs" / f"{name}_run-{seed}.log"
    )
    if reported is not None:
        value, per_atom = reported
        ours = saved["metrics"]["mae_per_atom_meV" if per_atom else "mae_meV"]
        # MACE prints one decimal of meV, so allow that rounding as well as 1 %.
        agree = abs(ours - value) <= max(0.01 * abs(value), 0.1)
        unit = "meV per atom" if per_atom else "meV"
        print(
            f"[results] MACE's own test table: {value:.5g} {unit} "
            + ("(agrees)" if agree else "(DIFFERS, check the saved files)"),
            flush=True,
        )
    return saved
