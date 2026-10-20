"""Train MACE on an energy-only OpenQDC dataset via extended XYZ files and ``mace_run_train``.

The OpenQDC data is first exported to ``{train,valid,test}.xyz`` (see
``export_xyz.py``; skipped if the files already exist), then the official MACE
training CLI is run on them with energy-only settings (no forces).

Examples
--------
Quick run on 10k QM9 molecules, energy column 0 (U0)::

    python scripts/openqdc/train_mace.py --dataset QM9 --energy-method 0 \
        --max-samples 10000 --epochs 50

Any extra ``mace_run_train`` flag is passed through unchanged, e.g.::

    python scripts/openqdc/train_mace.py --dataset Alchemy -- --num_channels=128 --max_L=1
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import torch
from export_xyz import (
    ENERGY_KEY,
    add_data_args,
    ensure_export,
    resolve_energy_arg,
)
from filelock import FileLock, Timeout


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_args(p)
    # Model
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--hidden-irreps", default="64x0e + 64x1o")
    p.add_argument("--max-ell", type=int, default=3)
    p.add_argument("--correlation", type=int, default=3)
    p.add_argument("--num-interactions", type=int, default=2)
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--valid-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--patience", type=int, default=30, help="early stopping (epochs)"
    )
    p.add_argument(
        "--dtype", default="float64", choices=["float32", "float64"]
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--error-table", default="TotalMAE")
    p.add_argument("--work-dir", default="logs/mace_openqdc")
    p.add_argument("--name", default=None, help="run name (default: derived)")
    return p.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    extra = [a for a in extra if a != "--"]
    args = resolve_energy_arg(args)

    paths = ensure_export(args)

    # Default run name: data settings (from the XYZ directory) plus a short hash
    # of the model/training settings, so runs that differ only in those never
    # share a folder.
    data_dir, split_dir = os.path.split(os.path.dirname(paths["train"]))
    keys = [
        "r_max",
        "hidden_irreps",
        "max_ell",
        "correlation",
        "num_interactions",
        "epochs",
        "batch_size",
        "valid_batch_size",
        "lr",
        "patience",
        "dtype",
        "seed",
        "error_table",
    ]
    settings = json.dumps(
        [{k: getattr(args, k) for k in keys}, extra], sort_keys=True
    )
    digest = hashlib.sha1(settings.encode()).hexdigest()[:8]
    name = args.name or f"{os.path.basename(data_dir)}_{split_dir}_{digest}"
    work_dir = os.path.join(args.work_dir, name)

    mace_run_train = shutil.which("mace_run_train") or os.path.join(
        os.path.dirname(sys.executable), "mace_run_train"
    )
    cmd = [
        mace_run_train,
        f"--name={name}",
        f"--work_dir={work_dir}",
        f"--seed={args.seed}",
        f"--device={args.device}",
        f"--default_dtype={args.dtype}",
        # Data: energy only, read from the XYZ files
        f"--train_file={paths['train']}",
        f"--valid_file={paths['valid']}",
        f"--test_file={paths['test']}",
        f"--energy_key={ENERGY_KEY}",
        "--E0s=average",  # per-element reference energies fitted on train
        # MACE has no energy-only loss: forces are predicted but get zero weight
        # (frames without REF_forces also get config forces_weight=0).
        "--loss=weighted",
        "--energy_weight=1.0",
        "--forces_weight=0.0",
        "--stage_two_forces_weight=0.0",
        "--scaling=std_scaling",  # rms_forces_scaling needs forces
        f"--error_table={args.error_table}",
        # Model
        "--model=MACE",
        f"--r_max={args.r_max}",
        f"--hidden_irreps={args.hidden_irreps}",
        f"--max_ell={args.max_ell}",
        f"--correlation={args.correlation}",
        f"--num_interactions={args.num_interactions}",
        # Training
        f"--max_num_epochs={args.epochs}",
        f"--patience={args.patience}",
        f"--batch_size={args.batch_size}",
        f"--valid_batch_size={args.valid_batch_size}",
        f"--lr={args.lr}",
        "--amsgrad",
        "--ema",
        "--ema_decay=0.99",
        *extra,
    ]
    print(" \\\n  ".join(cmd))
    os.makedirs(work_dir, exist_ok=True)
    # Refuse to start if another run is using this folder right now.
    lock = FileLock(os.path.join(work_dir, "run.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        sys.exit(
            f"Another run is already using {work_dir}. "
            "Give this run its own --name."
        )
    try:
        subprocess.run(cmd, check=True)
    finally:
        lock.release()
    print(f"Results, checkpoints and logs in {work_dir}")


if __name__ == "__main__":
    main()
