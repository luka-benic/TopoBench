"""Export an energy-only OpenQDC dataset to extended XYZ files for MACE.

Works for every OpenQDC potential-energy dataset. Each frame stores positions
(Angstrom), atomic numbers, ``pbc=F`` and one energy (eV) in
``info["REF_energy"]``; no forces are written. Energies and positions are the
values OpenQDC returns for ``energy_unit="ev"`` and ``distance_unit="ang"``.

Example
-------
::

    python scripts/openqdc/export_xyz.py --dataset QM9 --energy-method 0

The molecules are subsampled the same way as TopoBench's ``OpenQDCDataset``
(same ``--max-samples`` / ``--subsample-seed``).
"""

import argparse
import json
import logging
import os

import numpy as np
from ase import Atoms
from ase.io import write
from generate_configs import configured_energy_method
from tqdm import tqdm

from topobench.data.datasets.openqdc_dataset import PROCESSING_VERSION
from topobench.data.utils.openqdc_utils import (
    PROJECT_ROOT,
    OpenQDCSource,
    canonical_name,
    default_cache_dir,
    is_cached,
    locked,
    method_names_for,
    resolve_energy_method,
)

log = logging.getLogger(__name__)
ENERGY_KEY = "REF_energy"
SPLITS = ("train", "valid", "test")


def add_data_args(p: argparse.ArgumentParser) -> None:
    """Add the dataset/split arguments shared with ``train_mace.py``."""
    p.add_argument(
        "--dataset", required=True, help="OpenQDC dataset, e.g. QM9"
    )
    p.add_argument("--cache-dir", default=default_cache_dir())
    p.add_argument(
        "--xyz-root", default=str(PROJECT_ROOT / "datasets" / "openqdc_xyz")
    )
    p.add_argument(
        "--target",
        default="formation_energies",
        choices=["formation_energies", "energies"],
        help="Energy written to REF_energy (eV)",
    )
    p.add_argument(
        "--energy-method",
        default=None,
        help="energy column, index or name; default: the one chosen in the "
        "dataset yaml (loader.parameters.energy_method), else 0",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument("--valid-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument(
        "--allow-non-total",
        action="store_true",
        help="export datasets whose energies are not total energies "
        "(MACE learns total energies)",
    )


def resolve_energy_arg(args: argparse.Namespace) -> argparse.Namespace:
    """Turn ``--energy-method`` (index, name or None) into a column index.

    None means the column chosen in the dataset yaml, or 0 if there is none.
    """
    spec = args.energy_method
    if spec is None:
        spec = configured_energy_method(args.dataset)
    args.energy_method = resolve_energy_method(
        method_names_for(args.dataset), 0 if spec is None else spec
    )
    return args


def xyz_dir(args: argparse.Namespace) -> str:
    """Directory holding the XYZ files for these export settings."""
    n = args.max_samples if args.max_samples is not None else "all"
    name = "_".join(
        [
            f"v{PROCESSING_VERSION}",
            canonical_name(args.dataset),
            args.target,
            f"m{args.energy_method}",
            f"n{n}",
            f"s{args.subsample_seed}",
        ]
    )
    return os.path.join(
        args.xyz_root,
        name,
        f"split_v{args.valid_fraction}_t{args.test_fraction}_s{args.split_seed}",
    )


def export_marker(args: argparse.Namespace) -> str:
    """Path of the file written last by an export (it marks it as complete)."""
    return os.path.join(xyz_dir(args), "export_args.json")


def ensure_export(args: argparse.Namespace) -> dict[str, str]:
    """Return the XYZ paths, exporting first if no complete export exists.

    The export runs under a lock, so jobs started together wait for one
    export instead of writing the same files at the same time.

    Returns
    -------
    dict[str, str]
        Paths of the train/valid/test XYZ files.
    """
    os.makedirs(xyz_dir(args), exist_ok=True)
    with locked(os.path.join(xyz_dir(args), "export.lock"), "XYZ export"):
        if os.path.exists(export_marker(args)):
            print(f"[prepare] {args.dataset}: XYZ files already exported.")
        else:
            downloaded = is_cached(args.dataset, args.cache_dir)
            print(
                f"[prepare] {args.dataset}: OpenQDC data "
                + (
                    "already downloaded."
                    if downloaded
                    else "not found locally, downloading ..."
                )
                + " Exporting XYZ files ..."
            )
            export(args)
    return xyz_paths(args)


def xyz_paths(args: argparse.Namespace) -> dict[str, str]:
    """Paths of the train/valid/test XYZ files."""
    return {s: os.path.join(xyz_dir(args), f"{s}.xyz") for s in SPLITS}


def export(args: argparse.Namespace) -> dict[str, str]:
    """Read OpenQDC, split it and write the XYZ files.

    Returns
    -------
    dict[str, str]
        Paths of the train/valid/test XYZ files.
    """
    src = OpenQDCSource(args.dataset, args.cache_dir)
    if not 0 <= args.energy_method < src.n_methods:
        raise ValueError(
            f"--energy-method must be in [0, {src.n_methods - 1}] "
            f"for {src.name}: {src.method_names}"
        )
    if src.not_total_reason and not args.allow_non_total:
        raise SystemExit(
            f"{src.name} energies are not total energies "
            f"({src.not_total_reason}); MACE learns total energies. "
            "Use --allow-non-total to export anyway."
        )
    print(
        f"{src.name}: method {args.energy_method} = "
        f"{src.method_names[args.energy_method]!r} (eV, Angstrom)"
    )

    idxs = np.arange(len(src))
    if args.max_samples is not None and args.max_samples < len(src):
        rng = np.random.default_rng(args.subsample_seed)
        idxs = np.sort(rng.choice(len(src), args.max_samples, replace=False))
    elif len(idxs) > 500_000:
        log.warning(
            "Exporting %d molecules needs a lot of memory; consider --max-samples.",
            len(idxs),
        )

    frames = []
    for i in tqdm(idxs, desc=f"Reading {src.name}"):
        numbers, _, positions, energy = src.get(
            i, args.target, args.energy_method
        )
        if not np.isfinite(energy):
            continue  # energy not computed with this method
        atoms = Atoms(numbers=numbers, positions=positions, pbc=False)
        atoms.info[ENERGY_KEY] = energy
        atoms.info["openqdc_index"] = int(i)
        frames.append(atoms)

    perm = np.random.default_rng(args.split_seed).permutation(len(frames))
    n_valid = int(args.valid_fraction * len(frames))
    n_test = int(args.test_fraction * len(frames))
    split_idx = {
        "valid": perm[:n_valid],
        "test": perm[n_valid : n_valid + n_test],
        "train": perm[n_valid + n_test :],
    }

    paths = xyz_paths(args)
    os.makedirs(xyz_dir(args), exist_ok=True)
    for split, idx in split_idx.items():
        write(paths[split], [frames[j] for j in idx], format="extxyz")
        print(f"{split}: {len(idx)} frames -> {paths[split]}")
    with open(export_marker(args), "w") as f:
        json.dump(
            {
                **vars(args),
                "method_name": src.method_names[args.energy_method],
            },
            f,
            indent=2,
        )
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_args(p)
    p.add_argument("--overwrite", action="store_true")
    args = resolve_energy_arg(p.parse_args())
    if args.overwrite and os.path.exists(export_marker(args)):
        os.remove(export_marker(args))
    existed = os.path.exists(export_marker(args))
    ensure_export(args)
    if existed:
        print(f"XYZ files already exist in {xyz_dir(args)} (use --overwrite).")


if __name__ == "__main__":
    main()
