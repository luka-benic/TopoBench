"""Tests for the names and the versioning of the MACE .xyz exports."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("ase")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "openqdc"


@pytest.fixture
def export_xyz(monkeypatch):
    """Import scripts/openqdc/export_xyz.py (it is not a package)."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "export_xyz_under_test", SCRIPTS / "export_xyz.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def make_args(tmp_path, **overrides):
    """Return arguments like those of the export scripts."""
    values = {
        "dataset": "qm9",
        "target": "energies",
        "energy_method": 0,
        "max_samples": 100,
        "subsample_seed": 0,
        "valid_fraction": 0.1,
        "test_fraction": 0.1,
        "split_seed": 0,
        "xyz_root": str(tmp_path),
        "cache_dir": str(tmp_path / "cache"),
        "allow_non_total": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_total_energy_is_the_default_target(export_xyz):
    """The shared data arguments default to the total energy."""
    parser = argparse.ArgumentParser()
    export_xyz.add_data_args(parser)
    args = parser.parse_args(["--dataset", "QM9"])
    assert args.target is None  # not given: the dataset yaml decides
    assert export_xyz.resolve_energy_arg(args).target == "energies"


def test_data_settings_default_to_the_dataset_yaml(export_xyz, monkeypatch):
    """max_samples, subsample_seed and target come from the yaml like the split."""
    yaml_values = {"max_samples": 500, "subsample_seed": 3, "split_seed": 2}
    monkeypatch.setattr(
        export_xyz,
        "configured_loader_parameter",
        lambda name, key: yaml_values.get(key),
    )
    parser = argparse.ArgumentParser()
    export_xyz.add_data_args(parser)
    args = export_xyz.resolve_energy_arg(
        parser.parse_args(["--dataset", "QM9", "--split-seed", "7"])
    )
    assert (args.max_samples, args.subsample_seed) == (500, 3)
    assert args.split_seed == 7  # a flag beats the yaml
    assert (args.valid_fraction, args.target) == (0.1, "energies")


def test_yaml_target_that_mace_cannot_learn_is_refused(
    export_xyz, monkeypatch
):
    monkeypatch.setattr(
        export_xyz,
        "configured_loader_parameter",
        lambda name, key: (
            "per_atom_formation_energies" if key == "target" else None
        ),
    )
    parser = argparse.ArgumentParser()
    export_xyz.add_data_args(parser)
    with pytest.raises(SystemExit, match="--target"):
        export_xyz.resolve_energy_arg(parser.parse_args(["--dataset", "QM9"]))


def test_non_total_energies_are_refused_even_for_an_existing_export(
    export_xyz, tmp_path
):
    args = make_args(tmp_path, dataset="QM8")
    Path(export_xyz.xyz_dir(args)).mkdir(parents=True)
    Path(export_xyz.export_marker(args)).write_text("{}")  # made earlier
    with pytest.raises(SystemExit, match="not total energies"):
        export_xyz.ensure_export(args)
    args.allow_non_total = True
    export_xyz.ensure_export(args)  # allowed explicitly: the export is reused


@pytest.mark.parametrize("dataset", ["QM9", "QM7b", "Alchemy"])
def test_export_folder_names_include_only_required_unit_revision(
    export_xyz, tmp_path, dataset
):
    folder = Path(export_xyz.xyz_dir(make_args(tmp_path, dataset=dataset)))
    suffix = "_kcalmol_ev_v1" if dataset in ("QM9", "QM7b") else ""
    assert folder.parent.name == f"{dataset}_energies_m0_n100_s0{suffix}"
    assert folder.name == "split_v0.1_t0.1_s0"


def test_old_qm9_export_is_not_reused(export_xyz, tmp_path, monkeypatch):
    args = make_args(tmp_path)
    old = tmp_path / "QM9_energies_m0_n100_s0" / "split_v0.1_t0.1_s0"
    old.mkdir(parents=True)
    (old / "export_args.json").write_text("{}")
    (old / "train.xyz").write_text("old incorrectly scaled energies")
    calls = []
    monkeypatch.setattr(export_xyz, "is_cached", lambda *a, **k: True)
    monkeypatch.setattr(export_xyz, "export", lambda args: calls.append(args))
    paths = export_xyz.ensure_export(args)
    assert calls == [args]
    assert Path(paths["train"]).parent != old
    assert (old / "train.xyz").read_text() == "old incorrectly scaled energies"


def test_finished_export_is_not_made_again(export_xyz, tmp_path, monkeypatch):
    """An export with its marker file is reused; without it, it is made."""
    args = make_args(tmp_path)
    calls = []

    def fake_export(arguments):
        calls.append(1)
        with open(export_xyz.export_marker(arguments), "w") as handle:
            json.dump({}, handle)

    monkeypatch.setattr(export_xyz, "export", fake_export)
    monkeypatch.setattr(export_xyz, "is_cached", lambda *a, **k: True)

    export_xyz.ensure_export(args)
    export_xyz.ensure_export(args)
    assert len(calls) == 1  # the second call finds the finished export

    Path(export_xyz.export_marker(args)).unlink()
    export_xyz.ensure_export(args)
    assert len(calls) == 2  # an export without its marker is made again
