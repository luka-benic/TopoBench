"""Train MACE on an energy-only OpenQDC dataset via extended XYZ files and ``mace_run_train``.

The OpenQDC data is first exported to ``{train,valid,test}.xyz`` (see
``export_xyz.py``; skipped if the files already exist), then MACE's training
is run on them through ``run_mace_train.py`` with energy-only settings: forces
are not computed (``--compute_forces=False``) and the force terms of the loss
are skipped.

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
import subprocess
import sys
import uuid

import torch
from export_xyz import (
    ENERGY_KEY,
    add_data_args,
    ensure_export,
    resolve_energy_arg,
)
from filelock import FileLock, Timeout
from mace_test_results import (
    log_mace_results_to_wandb,
    save_mace_test_results,
)

from topobench.data.utils.openqdc_utils import canonical_name


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
    # Metrics: the loss that is minimised (MACE also picks the best checkpoint
    # by the validation loss) and the error table printed during/after training.
    p.add_argument(
        "--loss",
        default="weighted",
        choices=["weighted", "huber", "l1l2energyforces"],
        help="energy loss: weighted = mean squared error of the per-atom "
        "energy, l1l2energyforces = mean absolute error of the per-atom "
        "energy, huber = Huber loss (see --huber-delta; MACE then also "
        "computes stress, which removes the speed-up of not computing forces)",
    )
    p.add_argument("--huber-delta", type=float, default=0.01)
    p.add_argument(
        "--error-table",
        default="TotalMAE",
        choices=["TotalMAE", "PerAtomMAE", "TotalRMSE", "PerAtomRMSE"],
        help="metric shown in the validation log lines and the final tables",
    )
    # Weights & Biases (login once with `wandb login`, or set WANDB_API_KEY;
    # WANDB_MODE=offline logs to disk without network).
    p.add_argument(
        "--wandb", action="store_true", help="log the run to Weights & Biases"
    )
    p.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "openqdc"),
        help="project (default: $WANDB_PROJECT, else openqdc)",
    )
    p.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", ""),
        help="team or user (default: $WANDB_ENTITY, else your default)",
    )
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", nargs="+", default=None)
    p.add_argument(
        "--wandb-name", default=None, help="run name; default: the run name"
    )
    p.add_argument("--work-dir", default="logs/mace_openqdc")
    p.add_argument("--name", default=None, help="run name (default: derived)")
    p.add_argument(
        "--no-test-results",
        action="store_true",
        help="do not save the test-set predictions (.txt) and parity plot",
    )
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
        "loss",
        "huber_delta",
    ]
    # "compute_forces": False keeps these runs apart from older runs that
    # computed forces.
    settings = json.dumps(
        [
            {k: getattr(args, k) for k in keys},
            extra,
            {"compute_forces": False},
        ],
        sort_keys=True,
    )
    digest = hashlib.sha1(settings.encode()).hexdigest()[:8]
    name = args.name or f"{os.path.basename(data_dir)}_{split_dir}_{digest}"
    work_dir = os.path.join(args.work_dir, name)

    # MACE is always started through run_mace_train.py with this interpreter, so
    # training and the test evaluation use the same MACE (the one of this env).
    launcher = os.path.join(os.path.dirname(__file__), "run_mace_train.py")
    cmd = [
        sys.executable,
        launcher,
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
        # Energy only: forces are not computed (the force terms of the loss are
        # skipped by the launcher), frames without REF_forces get a forces
        # weight of 0 anyway, and the force weight is 0 as a second safeguard.
        "--compute_forces=False",
        f"--loss={args.loss}",
        "--energy_weight=1.0",
        "--forces_weight=0.0",
        "--scaling=std_scaling",  # the default rms_forces_scaling needs forces
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
    ]
    if args.loss == "huber":
        cmd.append(f"--huber_delta={args.huber_delta}")
    env = dict(os.environ)
    wandb_run_id = None
    if args.wandb:
        # The run id is fixed here so the test-set results can be added to the
        # same Weights & Biases run after mace_run_train has finished.
        wandb_run_id = uuid.uuid4().hex[:8]
        env["WANDB_RUN_ID"] = wandb_run_id
        env["WANDB_RESUME"] = "allow"
        # mace_run_train has no group/tag options; wandb reads these variables.
        if args.wandb_group:
            env["WANDB_RUN_GROUP"] = args.wandb_group
        tags = args.wandb_tags or [canonical_name(args.dataset), "MACE"]
        env["WANDB_TAGS"] = ",".join(tags)
        cmd += [
            "--wandb",
            f"--wandb_project={args.wandb_project}",
            f"--wandb_entity={args.wandb_entity}",
            f"--wandb_name={args.wandb_name or name}",
            f"--wandb_dir={os.path.abspath(work_dir)}",
            "--wandb_log_hypers",
            *[
                "hidden_irreps",
                "r_max",
                "max_ell",
                "correlation",
                "num_interactions",
                "lr",
                "batch_size",
                "max_num_epochs",
                "loss",
                "huber_delta",
                "seed",
            ],
        ]
    cmd += extra
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
        subprocess.run(cmd, check=True, env=env)
    finally:
        lock.release()
    print(f"Results, checkpoints and logs in {work_dir}")
    if not args.no_test_results:
        # Predictions of the best model on test.xyz: .txt files + parity plot.
        saved = save_mace_test_results(
            work_dir,
            name,
            paths["test"],
            canonical_name(args.dataset),
            args.energy_method,
            args.device,
            args.valid_batch_size,
        )
        if args.wandb:
            log_mace_results_to_wandb(
                saved,
                wandb_run_id,
                args.wandb_project,
                args.wandb_entity,
                os.path.abspath(work_dir),
                {
                    "dataset": canonical_name(args.dataset),
                    "energy_method": args.energy_method,
                    "max_samples": args.max_samples,
                    "target": args.target,
                },
            )


if __name__ == "__main__":
    main()
