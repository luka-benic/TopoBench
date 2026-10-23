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
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_total_energy_is_the_default_target(export_xyz):
    """The shared data arguments default to the total energy."""
    parser = argparse.ArgumentParser()
    export_xyz.add_data_args(parser)
    assert parser.parse_args(["--dataset", "QM9"]).target == "energies"


def test_export_folder_name_has_no_version_prefix(export_xyz, tmp_path):
    """The folder name is readable and does not start with a version."""
    folder = Path(export_xyz.xyz_dir(make_args(tmp_path)))
    assert folder.parent.name == "QM9_energies_m0_n100_s0"
    assert folder.name == "split_v0.1_t0.1_s0"


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
