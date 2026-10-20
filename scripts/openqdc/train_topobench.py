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
import os
import subprocess
import sys

from generate_configs import configured_energy_method, ensure_config

from topobench.data.utils.openqdc_utils import (
    PROJECT_ROOT,
    canonical_name,
    default_cache_dir,
    method_names_for,
    resolve_energy_method,
)


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
            "formation_energies",
            "energies",
            "per_atom_formation_energies",
        ],
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--subsample-seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--logger", default="csv", help="csv, wandb, tensorboard, ..."
    )
    p.add_argument("--cache-dir", default=default_cache_dir())
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
    method = None
    if args.energy_method is not None:
        method = resolve_energy_method(names, args.energy_method)
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

    params = "dataset.loader.parameters"
    overrides = [
        f"dataset=graph/{config.stem}",
        f"model={args.model}",
        f"logger={args.logger}",
    ]
    if method is not None:
        overrides.append(f"{params}.energy_method={method}")
    for key, value in [
        (f"{params}.target", args.target),
        (f"{params}.max_samples", args.max_samples),
        (f"{params}.subsample_seed", args.subsample_seed),
        ("trainer.max_epochs", args.epochs),
        ("dataset.dataloader_params.batch_size", args.batch_size),
        ("seed", args.seed),
    ]:
        if value is not None:
            overrides.append(f"{key}={value}")

    cmd = [sys.executable, "-m", "topobench", *overrides, *extra]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
