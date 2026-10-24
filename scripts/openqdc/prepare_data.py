"""Download and preprocess an OpenQDC dataset before training.

Checks whether the dataset is downloaded and preprocessed for the given
settings and does whatever is missing. ``train_topobench.py`` and
``train_mace.py`` call this automatically before they start training; it can
also be run on its own, for example once before launching several jobs.

* ``topobench``: download the OpenQDC data, then convert it to the graphs the
  training run will use (same yaml, same overrides).
* ``mace``: download the OpenQDC data, then export the ``.xyz`` files.

Examples
--------
::

    python scripts/openqdc/prepare_data.py topobench --dataset QM9 --energy-method 3
    python scripts/openqdc/prepare_data.py topobench --dataset QM9 --max-samples 20000 \
        --loader-param baseline=none
    python scripts/openqdc/prepare_data.py mace --dataset QM9 --max-samples 20000
"""

import argparse
import os
from typing import Any

import yaml
from export_xyz import add_data_args, ensure_export, resolve_energy_arg
from generate_configs import ensure_config
from omegaconf import OmegaConf

from topobench.data.loaders.graph import OpenQDCDatasetLoader
from topobench.data.utils.openqdc_utils import (
    PROJECT_ROOT,
    canonical_name,
    default_cache_dir,
    is_cached,
    method_names_for,
    resolve_energy_method,
)


def parse_loader_params(items: list[str], bare: bool = True) -> dict[str, Any]:
    """Parse ``key=value`` strings (as in Hydra overrides) into a dict.

    Parameters
    ----------
    items : list[str]
        Strings such as ``"baseline=none"`` or
        ``"dataset.loader.parameters.max_samples=20000"``; other strings are
        ignored.
    bare : bool
        Accept keys without a dot (``"baseline=none"``) as loader parameters.
        Pass False for Hydra overrides, whose root-level keys (``trainer=cpu``,
        ``seed=1``) are not loader parameters.

    Returns
    -------
    dict[str, Any]
        Loader parameter name to value (YAML-typed).
    """
    prefix = "dataset.loader.parameters."
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            continue
        key = key.lstrip("+")  # +key / ++key are Hydra's add / force-add
        if key.startswith(prefix):
            key = key[len(prefix) :]
        elif "." in key or not bare:
            continue  # an override of something else (model, trainer, ...)
        out[key] = yaml.safe_load(value)
    return out


def prepare_topobench(
    name: str,
    cache_dir: str,
    energy_method=None,
    overrides: dict[str, Any] | None = None,
) -> str:
    """Make sure the TopoBench graphs of a dataset exist, creating them if not.

    Uses the dataset yaml with the same overrides a training run would use, so
    the graphs created here are exactly the ones the run then loads.

    Parameters
    ----------
    name : str
        Dataset name.
    cache_dir : str
        OpenQDC cache directory.
    energy_method : int or str, optional
        Energy column; None keeps the one in the yaml.
    overrides : dict, optional
        Loader parameters overriding the yaml (None values are ignored).

    Returns
    -------
    str
        Folder with the converted graphs.
    """
    name = canonical_name(name)
    os.environ["OPENQDC_CACHE_DIR"] = str(cache_dir)
    config = ensure_config(name, cache_dir, energy_method=energy_method)
    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": str(PROJECT_ROOT / "datasets")},
            "dataset": OmegaConf.load(config),
        }
    )
    values = {k: v for k, v in (overrides or {}).items() if v is not None}
    if energy_method is not None:
        names = method_names_for(name)
        values["energy_method"] = resolve_energy_method(names, energy_method)
    for key, value in values.items():
        OmegaConf.update(cfg, f"dataset.loader.parameters.{key}", value)
    params = OmegaConf.create(
        OmegaConf.to_container(cfg.dataset.loader.parameters, resolve=True)
    )

    downloaded = is_cached(name, cache_dir)
    print(
        f"[prepare] {name}: checking OpenQDC data and graphs ...", flush=True
    )
    dataset, data_dir = OpenQDCDatasetLoader(params).load()
    if dataset.was_processed:
        print(
            f"[prepare] {name}: OpenQDC data "
            + ("was already downloaded" if downloaded else "downloaded now")
            + f"; graphs preprocessed now ({len(dataset)} graphs) in {data_dir}"
        )
    else:
        print(
            f"[prepare] {name}: graphs already preprocessed "
            f"({len(dataset)} graphs) in {data_dir}"
        )
    return data_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="pipeline", required=True)

    t = sub.add_parser("topobench", help="download + convert to graphs")
    t.add_argument("--dataset", required=True)
    t.add_argument("--cache-dir", default=default_cache_dir())
    t.add_argument("--energy-method", default=None)
    t.add_argument("--target", default=None)
    t.add_argument("--max-samples", type=int, default=None)
    t.add_argument("--subsample-seed", type=int, default=None)
    t.add_argument("--split-seed", type=int, default=None)
    t.add_argument("--valid-fraction", type=float, default=None)
    t.add_argument("--test-fraction", type=float, default=None)
    t.add_argument(
        "--loader-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="any other loader parameter of the yaml, e.g. baseline=none",
    )

    m = sub.add_parser("mace", help="download + export .xyz files")
    add_data_args(m)

    args = p.parse_args()
    if args.pipeline == "topobench":
        overrides = parse_loader_params(args.loader_param)
        overrides.update(
            {
                "target": args.target,
                "max_samples": args.max_samples,
                "subsample_seed": args.subsample_seed,
                "split_seed": args.split_seed,
                "valid_fraction": args.valid_fraction,
                "test_fraction": args.test_fraction,
            }
        )
        prepare_topobench(
            args.dataset, args.cache_dir, args.energy_method, overrides
        )
    else:
        ensure_export(resolve_energy_arg(args))


if __name__ == "__main__":
    main()
