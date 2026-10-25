"""Train a TopoBench model on any energy-only OpenQDC dataset.

The dataset config (``configs/dataset/graph/openqdc_<name>.yaml``) is created
on demand by ``generate_configs.py`` -- downloading the dataset to detect its
elements and units -- and then ``python -m topobench`` is run on it. Anything
after ``--`` is passed on as extra Hydra overrides.

Examples
--------
::

    python scripts/openqdc/train_topobench.py --dataset QM9 --model graph/gcn \
        --energy-method 0 --max-samples 20000 --epochs 50
    python scripts/openqdc/train_topobench.py --dataset Alchemy --model cell/cwn \
        -- optimizer.parameters.lr=0.001
"""

import argparse
import json
import os
import re
import subprocess
import sys

from generate_configs import configured_energy_method, ensure_config
from omegaconf import OmegaConf
from prepare_data import parse_loader_params, prepare_topobench

from topobench.data.utils.openqdc_utils import (
    PROJECT_ROOT,
    canonical_name,
    default_cache_dir,
    method_names_for,
    resolve_energy_method,
)

PER_ATOM = "per_atom_formation_energies"


def hydra_value(value: object) -> str:
    """Format a value for a Hydra command-line override.

    Parameters
    ----------
    value : object
        The value; strings with spaces or special characters get quoted.

    Returns
    -------
    str
        The text to put after ``key=``.
    """
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
        return text
    return json.dumps(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--dataset", required=True, help="OpenQDC dataset, e.g. QM9"
    )
    p.add_argument(
        "--model",
        default="graph/gcn",
        help="TopoBench model config, e.g. graph/gcn, simplicial/scn, cell/cwn",
    )
    p.add_argument(
        "--energy-method",
        default=None,
        help="energy column, index or name; default: the one chosen in the "
        "dataset yaml (loader.parameters.energy_method)",
    )
    p.add_argument(
        "--target",
        default=None,
        choices=[
            "energies",
            "formation_energies",
            "per_atom_formation_energies",
        ],
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--subsample-seed", type=int, default=None)
    p.add_argument(
        "--baseline",
        choices=["none", "per_element"],
        default=None,
        help="none: train on the raw energy; per_element: subtract per-element "
        "energies fitted on the training split. Default: the dataset yaml "
        "(none).",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="seed of the train/valid/test split (shared with MACE)",
    )
    p.add_argument("--valid-fraction", type=float, default=None)
    p.add_argument("--test-fraction", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--logger", default="csv", help="csv, wandb, tensorboard, ..."
    )
    # Metrics: which loss is minimised, which metric picks the best checkpoint
    # (and stops training early), and which metrics are reported.
    p.add_argument(
        "--loss",
        choices=["mse", "mae"],
        default=None,
        help="training loss; default: loss_type of the dataset yaml (mse)",
    )
    p.add_argument(
        "--monitor-metric",
        choices=["mae", "mse", "rmse"],
        default=None,
        help="validation metric that selects the best checkpoint and stops "
        "training early; default: monitor_metric of the yaml (mae)",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=["mae", "mse", "rmse", "r2"],
        default=None,
        help="metrics reported for validation and test; default: mae mse",
    )
    # Weights & Biases (login once with `wandb login`, or set WANDB_API_KEY;
    # WANDB_MODE=offline logs to disk without network).
    p.add_argument(
        "--wandb",
        action="store_true",
        help="log the run to Weights & Biases (same as --logger wandb)",
    )
    p.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "openqdc"),
        help="project (default: $WANDB_PROJECT, else openqdc)",
    )
    p.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="team or user (default: $WANDB_ENTITY, else your default)",
    )
    p.add_argument(
        "--wandb-name", default=None, help="run name; default: derived"
    )
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", nargs="+", default=None)
    p.add_argument("--cache-dir", default=default_cache_dir())
    p.add_argument(
        "--no-test-results",
        action="store_true",
        help="do not save the test-set predictions (.txt) and parity plot",
    )
    p.add_argument(
        "--regen-config",
        action="store_true",
        help="re-detect elements/units and rewrite the dataset yaml",
    )
    args, extra = p.parse_known_args()
    extra = [a for a in extra if a != "--"]

    name = canonical_name(args.dataset)
    os.environ["OPENQDC_CACHE_DIR"] = args.cache_dir
    names = method_names_for(name)
    # Hydra overrides given after the flags win, as they do in the training
    # run itself, so the data prepared below is the data the run uses.
    extra_params = parse_loader_params(extra, bare=False)
    method = None
    spec = extra_params.get("energy_method", args.energy_method)
    if spec is not None:
        method = resolve_energy_method(names, spec)
    # A new yaml gets the chosen column; an existing yaml keeps its own.
    config = ensure_config(
        name,
        args.cache_dir,
        regenerate=args.regen_config,
        energy_method=method,
    )
    print(f"Using {config}")
    column = method
    if column is None:
        column = resolve_energy_method(names, configured_energy_method(name))
    print(f"Energy column {column}: {names[column]!r}")

    # Check that the dataset is downloaded and preprocessed for exactly these
    # settings, and do whatever is missing, before training starts.
    loader_overrides = {
        "target": args.target,
        "max_samples": args.max_samples,
        "subsample_seed": args.subsample_seed,
        "baseline": args.baseline,
        "split_seed": args.split_seed,
        "valid_fraction": args.valid_fraction,
        "test_fraction": args.test_fraction,
    }
    loader_overrides.update(extra_params)
    prepare_topobench(name, args.cache_dir, method, loader_overrides)

    params = "dataset.loader.parameters"
    overrides = [
        f"dataset=graph/{config.stem}",
        f"model={args.model}",
        f"logger={'wandb' if args.wandb else args.logger}",
    ]
    if args.loss:
        overrides.append(f"dataset.parameters.loss_type={args.loss}")
    metrics = list(args.metrics) if args.metrics else None
    if args.monitor_metric:
        overrides.append(
            f"dataset.parameters.monitor_metric={args.monitor_metric}"
        )
        # The monitored metric must be among the reported ones.
        metrics = metrics or ["mae", "mse"]
        if args.monitor_metric not in metrics:
            metrics.append(args.monitor_metric)
    if metrics:
        overrides.append(f"+dataset.parameters.metrics=[{','.join(metrics)}]")
    if args.wandb:
        run_name = args.wandb_name or "_".join(
            [name, args.model.replace("/", "-"), f"m{column}"]
            + ([f"n{args.max_samples}"] if args.max_samples else [])
        )
        overrides += [
            f"logger.wandb.project={hydra_value(args.wandb_project)}",
            f"+logger.wandb.name={hydra_value(run_name)}",
        ]
        if args.wandb_entity:
            overrides.append(
                f"+logger.wandb.entity={hydra_value(args.wandb_entity)}"
            )
        if args.wandb_group:
            overrides.append(
                f"logger.wandb.group={hydra_value(args.wandb_group)}"
            )
        tags = args.wandb_tags or [name, args.model.replace("/", "-")]
        overrides.append(
            f"logger.wandb.tags=[{','.join(hydra_value(t) for t in tags)}]"
        )
    if method is not None:
        overrides.append(f"{params}.energy_method={method}")
    for key, value in [
        (f"{params}.target", args.target),
        (f"{params}.max_samples", args.max_samples),
        (f"{params}.subsample_seed", args.subsample_seed),
        (f"{params}.baseline", args.baseline),
        (f"{params}.split_seed", args.split_seed),
        (f"{params}.valid_fraction", args.valid_fraction),
        (f"{params}.test_fraction", args.test_fraction),
        ("trainer.max_epochs", args.epochs),
        ("dataset.dataloader_params.batch_size", args.batch_size),
        ("seed", args.seed),
    ]:
        if value is not None:
            overrides.append(f"{key}={value}")

    # The target of the run: a Hydra override, else the flag, else the yaml.
    target = args.target or OmegaConf.load(config).loader.parameters.target
    for item in extra:
        if item.startswith(f"{params}.target="):
            target = item.split("=", 1)[1]
    # Always saves the split of the run (when training starts); unless
    # --no-test-results it also saves the per-molecule test predictions (.txt)
    # and a parity plot when training ends, in <run folder>/test_results.
    callback = "callbacks.openqdc_test_results"
    overrides += [
        f"+{callback}._target_=topobench.callbacks.openqdc_test_results.OpenQDCTestResults",
        f"+{callback}.dataset={name}",
        f"+{callback}.model={args.model}",
        f"+{callback}.energy_method={column}",
        f"+{callback}.per_atom={str(target == PER_ATOM).lower()}",
        f"+{callback}.save_results={str(not args.no_test_results).lower()}",
    ]
    if target == PER_ATOM and not any("pooling_type" in a for a in extra):
        # The target is an intensive per-atom energy: average over the atoms
        # instead of summing, so the network does not have to learn 1/N.
        overrides.append("++model.readout.pooling_type=mean")
        print("Per-atom target: model.readout.pooling_type=mean")

    cmd = [sys.executable, "-m", "topobench", *overrides, *extra]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
