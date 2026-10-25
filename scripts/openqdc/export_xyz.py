"""Export an energy-only OpenQDC dataset to extended XYZ files for MACE.

Works for every OpenQDC potential-energy dataset. Each frame stores positions
(Angstrom), atomic numbers, ``pbc=F`` and one energy (eV) in
``info["REF_energy"]``; no forces are written. The shared ``OpenQDCSource``
corrects QM9/QM7b source-unit metadata before requesting eV and Angstrom.

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
from generate_configs import (
    configured_energy_method,
    configured_loader_parameter,
)
from tqdm import tqdm

from topobench.data.utils.openqdc_utils import (
    NOT_TOTAL_ENERGY,
    PROJECT_ROOT,
    OpenQDCSource,
    canonical_name,
    default_cache_dir,
    energy_cache_suffix,
    is_cached,
    locked,
    method_names_for,
    resolve_energy_method,
    split_positions,
)

log = logging.getLogger(__name__)
ENERGY_KEY = "REF_energy"
SPLITS = ("train", "valid", "test")
# Data settings that default to the dataset yaml, so that MACE and TopoBench
# see the same molecules and target unless told otherwise.
YAML_DEFAULTS = {
    "valid_fraction": 0.1,
    "test_fraction": 0.1,
    "split_seed": 0,
    "max_samples": None,
    "subsample_seed": 0,
    "target": "energies",
}
TARGETS = ("energies", "formation_energies")


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
        default=None,
        choices=list(TARGETS),
        help="Energy written to REF_energy (eV): energies = total energy; "
        "default: loader.parameters.target of the dataset yaml, else energies",
    )
    p.add_argument(
        "--energy-method",
        default=None,
        help="energy column, index or name; default: the one chosen in the "
        "dataset yaml (loader.parameters.energy_method), else 0",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="default: loader.parameters.max_samples of the dataset yaml "
        "(null = all molecules)",
    )
    p.add_argument(
        "--subsample-seed",
        type=int,
        default=None,
        help="default: loader.parameters.subsample_seed of the dataset yaml",
    )
    p.add_argument(
        "--valid-fraction",
        type=float,
        default=None,
        help="default: loader.parameters.valid_fraction of the dataset yaml",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="default: loader.parameters.test_fraction of the dataset yaml",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="default: loader.parameters.split_seed of the dataset yaml; "
        "TopoBench uses the same split",
    )
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
    for key, default in YAML_DEFAULTS.items():
        if getattr(args, key) is None:
            value = configured_loader_parameter(args.dataset, key)
            setattr(args, key, default if value is None else value)
    if args.target not in TARGETS:
        raise SystemExit(
            f"MACE learns total or formation energies, but target "
            f"'{args.target}' is set for {args.dataset} in its yaml; pass "
            f"--target {' or '.join(TARGETS)}."
        )
    return args


def xyz_dir(args: argparse.Namespace) -> str:
    """Directory holding the XYZ files for these export settings."""
    n = args.max_samples if args.max_samples is not None else "all"
    name = "_".join(
        [
            canonical_name(args.dataset),
            args.target,
            f"m{args.energy_method}",
            f"n{n}",
            f"s{args.subsample_seed}",
        ]
    ) + energy_cache_suffix(args.dataset)
    return os.path.join(
        args.xyz_root,
        name,
        f"split_v{args.valid_fraction}_t{args.test_fraction}_s{args.split_seed}",
    )


def export_marker(args: argparse.Namespace) -> str:
    """Path of the file written last by an export (it marks it as complete)."""
    return os.path.join(xyz_dir(args), "export_args.json")


def ensure_export(
    args: argparse.Namespace, overwrite: bool = False
) -> dict[str, str]:
    """Return the XYZ paths, exporting first if no complete export exists.

    The export runs under a lock, so jobs started together wait for one
    export instead of writing the same files at the same time.

    Parameters
    ----------
    args : argparse.Namespace
        Data arguments with the energy column resolved.
    overwrite : bool
        Export again even if a complete export exists.

    Returns
    -------
    dict[str, str]
        Paths of the train/valid/test XYZ files.
    """
    reason = NOT_TOTAL_ENERGY.get(canonical_name(args.dataset))
    if reason and not args.allow_non_total:
        # Checked before an existing export is reused too, so a folder made
        # with --allow-non-total does not bypass the refusal.
        raise SystemExit(
            f"{canonical_name(args.dataset)} energies are not total energies "
            f"({reason}); MACE learns total energies. "
            "Use --allow-non-total to export anyway."
        )
    os.makedirs(xyz_dir(args), exist_ok=True)
    with locked(os.path.join(xyz_dir(args), "export.lock"), "XYZ export"):
        if overwrite and os.path.exists(export_marker(args)):
            os.remove(export_marker(args))
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


def partition_frames(
    frames: list[Atoms],
    valid_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> dict[str, np.ndarray]:
    """Split frames with the split TopoBench uses (``split_molecules``).

    Parameters
    ----------
    frames : list[ase.Atoms]
        Frames with ``info["openqdc_index"]``.
    valid_fraction : float
        Fraction of the frames in the validation set.
    test_fraction : float
        Fraction of the frames in the test set.
    split_seed : int
        Seed of the split.

    Returns
    -------
    dict[str, numpy.ndarray]
        The positions in ``frames`` of each split.
    """
    return split_positions(
        [a.info["openqdc_index"] for a in frames],
        valid_fraction,
        test_fraction,
        split_seed,
    )


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

    split_idx = partition_frames(
        frames, args.valid_fraction, args.test_fraction, args.split_seed
    )

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
                # every element of the three files: MACE needs them all in its
                # element table (a test-only element would crash it)
                "atomic_numbers": sorted(
                    {int(z) for a in frames for z in a.numbers}
                ),
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
    existed = os.path.exists(export_marker(args)) and not args.overwrite
    ensure_export(args, overwrite=args.overwrite)
    if existed:
        print(f"XYZ files already exist in {xyz_dir(args)} (use --overwrite).")


if __name__ == "__main__":
    main()
