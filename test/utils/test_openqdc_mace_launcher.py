"""Tests for the energy-only patches of the MACE launcher."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
mace_loss = pytest.importorskip("mace.modules.loss")
wandb = pytest.importorskip("wandb")

LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "openqdc"
    / "run_mace_train.py"
)


@pytest.fixture
def launcher(monkeypatch):
    """Import the launcher; everything it patches is restored afterwards."""
    for name in ("mean_squared_error_forces", "mean_normed_error_forces"):
        monkeypatch.setattr(mace_loss, name, getattr(mace_loss, name))
    monkeypatch.setattr(wandb, "init", wandb.init)
    monkeypatch.setattr(
        wandb, "log", getattr(wandb, "log", None), raising=False
    )
    spec = importlib.util.spec_from_file_location("run_mace_train", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_force_term_is_zero_when_forces_are_not_computed(launcher):
    """The wrapped force loss returns a zero with the energy's dtype."""
    calls = []

    def original(ref, pred, ddp=None):
        calls.append(1)
        return torch.tensor(5.0)

    wrapped = launcher.skip_without_forces(original)
    ref = {"energy": torch.zeros(3, dtype=torch.float64)}
    out = wrapped(ref, {"energy": ref["energy"], "forces": None})
    assert out.item() == 0.0
    assert out.dtype == torch.float64
    assert out.dim() == 0
    assert not calls


def test_force_term_is_unchanged_when_forces_exist(launcher):
    """With predicted forces the original loss function is called."""
    wrapped = launcher.skip_without_forces(lambda ref, pred, ddp=None: 7)
    ref = {"energy": torch.zeros(2)}
    assert wrapped(ref, {"forces": torch.ones(2, 3)}) == 7


def test_launcher_patches_the_mace_force_losses(launcher):
    """Importing the launcher patches both force terms of MACE's losses."""
    for name in launcher.FORCE_TERMS:
        assert hasattr(getattr(mace_loss, name), "__wrapped__")
    ref = {"energy": torch.zeros(2, dtype=torch.float64)}
    out = mace_loss.mean_squared_error_forces(ref, {"forces": None})
    assert out.item() == 0.0


def test_missing_metric_placeholders_become_none(launcher):
    """MACE's placeholder for a missing metric is replaced, also when nested."""

    class NoneMultiply:
        pass

    logged = {"valid": {"rmse_e": 0.1, "rmse_f": NoneMultiply()}, "epoch": 3}
    assert launcher.clean_metric(logged) == {
        "valid": {"rmse_e": 0.1, "rmse_f": None},
        "epoch": 3,
    }


def test_stale_checkpoints_are_moved_aside_not_deleted(tmp_path):
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "scripts" / "openqdc")
    )
    from train_mace import archive_old_checkpoints

    folder = tmp_path / "checkpoints"
    folder.mkdir()
    (folder / "run_epoch-80.pt").write_text("old")
    archive_old_checkpoints(str(tmp_path), restarting=True)
    assert (folder / "run_epoch-80.pt").exists()  # --restart_latest wants them
    archive_old_checkpoints(str(tmp_path), restarting=False)
    assert not folder.exists()
    moved = list(tmp_path.glob("checkpoints_previous_*"))
    assert (
        len(moved) == 1 and (moved[0] / "run_epoch-80.pt").read_text() == "old"
    )
    archive_old_checkpoints(str(tmp_path), restarting=False)  # nothing to do


def test_element_list_of_an_export_is_read_from_its_marker(tmp_path):
    import json
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "scripts" / "openqdc")
    )
    from train_mace import exported_elements

    assert exported_elements(str(tmp_path / "train.xyz")) is None  # no marker
    (tmp_path / "export_args.json").write_text(
        json.dumps({"atomic_numbers": [1, 6, 9]})
    )
    assert exported_elements(str(tmp_path / "train.xyz")) == "[1, 6, 9]"
