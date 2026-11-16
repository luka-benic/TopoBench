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


def test_elements_of_an_export_come_from_its_marker_or_the_files(tmp_path):
    import json
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "scripts" / "openqdc")
    )
    from ase import Atoms
    from ase.io import write
    from train_mace import export_elements

    paths = {k: str(tmp_path / f"{k}.xyz") for k in ("train", "valid", "test")}
    write(
        paths["train"],
        [Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])],
    )
    write(paths["valid"], [Atoms("CH", positions=[[0, 0, 0], [1, 0, 0]])])
    write(paths["test"], [Atoms("N", positions=[[0, 0, 0]])])
    assert export_elements(paths) == [1, 6, 7, 8]  # no marker: read the files
    (tmp_path / "export_args.json").write_text(
        json.dumps({"atomic_numbers": [9, 1, 6]})
    )
    assert export_elements(paths) == [1, 6, 9]  # the marker wins


def test_baseline_choice_sets_the_e0s_of_mace(tmp_path):
    import ast
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "scripts" / "openqdc")
    )
    from train_mace import e0s_argument

    assert e0s_argument("per_element", [1, 6]) == "--E0s=average"
    none = e0s_argument("none", [1, 6, 8])
    assert none.startswith("--E0s=")
    assert ast.literal_eval(none[len("--E0s=") :]) == {1: 0.0, 6: 0.0, 8: 0.0}


@pytest.fixture
def train_mace(monkeypatch, tmp_path):
    """train_mace.py with everything heavy replaced; records the commands."""
    import subprocess
    import sys

    scripts = str(Path(__file__).resolve().parents[2] / "scripts" / "openqdc")
    monkeypatch.syspath_prepend(scripts)
    spec = importlib.util.spec_from_file_location(
        "train_mace_under_test", Path(scripts) / "train_mace.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    folder = tmp_path / "QM9_energies_m0_nall_s0" / "split_v0.1_t0.1_s0"
    folder.mkdir(parents=True)
    paths = {k: str(folder / f"{k}.xyz") for k in ("train", "valid", "test")}
    monkeypatch.setattr(module, "ensure_export", lambda args: paths)
    monkeypatch.setattr(module, "export_elements", lambda p: [1, 6, 8])
    monkeypatch.setattr(module, "save_mace_split", lambda *a, **k: None)
    monkeypatch.setattr(module, "save_mace_test_results", lambda *a, **k: {})
    commands, codes = [], []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        code = codes.pop(0) if codes else 0
        if code:
            raise subprocess.CalledProcessError(code, cmd)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    def run(argv, fail=()):
        codes[:] = list(fail)
        commands.clear()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_mace.py",
                "--dataset",
                "QM9",
                "--device",
                "cpu",
                "--work-dir",
                str(tmp_path / "runs"),
                *argv,
            ],
        )
        module.main()
        return list(commands)

    yield run
    sys.modules.pop(spec.name, None)


def option(cmd, prefix):
    return next(a for a in cmd if a.startswith(prefix)).split("=", 1)[1]


def test_mace_seeds_are_trained_one_after_another_in_their_own_folders(
    train_mace,
):
    commands = train_mace(["--seed", "0", "1", "2"])
    assert [option(c, "--seed=") for c in commands] == ["0", "1", "2"]
    folders = [option(c, "--work_dir=") for c in commands]
    assert len(set(folders)) == 3  # the run name has a hash of the seed
    assert (
        train_mace(["--seed", "7"])
        and option(train_mace(["--seed", "7"])[0], "--seed=") == "7"
    )


def test_mace_explicit_names_get_the_seed_when_several_are_trained(
    train_mace,
):
    several = train_mace(["--seed", "0", "1", "--name", "exp"])
    assert [option(c, "--name=") for c in several] == [
        "exp_seed0",
        "exp_seed1",
    ]
    single = train_mace(["--seed", "3", "--name", "exp"])
    assert option(single[0], "--name=") == "exp"


def test_mace_failing_seed_does_not_stop_the_others(train_mace):
    with pytest.raises(SystemExit) as stop:
        train_mace(["--seed", "0", "1", "2"], fail=[0, 5, 0])
    assert stop.value.code == 5
