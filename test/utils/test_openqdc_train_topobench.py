"""Tests of training several models in one call of train_topobench.py."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "openqdc"
YAML = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "dataset"
    / "graph"
    / "openqdc_qm9.yaml"
)


@pytest.fixture
def script(monkeypatch):
    """Import scripts/openqdc/train_topobench.py (it is not a package)."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "train_topobench_under_test", SCRIPTS / "train_topobench.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ensure_config", lambda *a, **k: YAML)
    monkeypatch.setattr(module, "prepare_topobench", lambda *a, **k: None)
    yield module
    sys.modules.pop(spec.name, None)


def run_main(script, monkeypatch, argv, codes=None):
    """Run main() with a fake subprocess; return the commands it started."""
    commands = []
    codes = list(codes or [])

    real_run = script.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if "topobench" not in cmd:  # only the training command is faked
            return real_run(cmd, *args, **kwargs)
        commands.append(cmd)
        return SimpleNamespace(returncode=codes.pop(0) if codes else 0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["train_topobench.py", "--dataset", "QM9", *argv]
    )
    script.main()
    return commands


def value_of(cmd, prefix):
    return next(a for a in cmd if a.startswith(prefix)).split("=", 1)[1]


def test_models_are_trained_one_after_another_in_the_given_order(
    script, monkeypatch
):
    commands = run_main(
        script, monkeypatch, ["--model", "graph/gcn", "graph/gin", "cell/cwn"]
    )
    assert [value_of(c, "model=") for c in commands] == [
        "graph/gcn",
        "graph/gin",
        "cell/cwn",
    ]
    # the result callback names the model of its own run
    assert [
        value_of(c, "+callbacks.openqdc_test_results.model=") for c in commands
    ] == ["graph/gcn", "graph/gin", "cell/cwn"]


def test_default_is_one_model_and_repeated_names_run_once(script, monkeypatch):
    assert len(run_main(script, monkeypatch, [])) == 1
    commands = run_main(
        script, monkeypatch, ["--model", "graph/gcn", "graph/gcn", "graph/gin"]
    )
    assert [value_of(c, "model=") for c in commands] == [
        "graph/gcn",
        "graph/gin",
    ]


def test_a_failing_model_does_not_stop_the_others(script, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run_main(
            script,
            monkeypatch,
            ["--model", "graph/gcn", "graph/gin", "cell/cwn"],
            codes=[0, 3, 0],
        )
    assert stop.value.code == 3  # the exit code of the first failure


def test_wandb_names_get_the_model_when_several_models_are_trained(
    script, monkeypatch
):
    several = run_main(
        script,
        monkeypatch,
        [
            "--model",
            "graph/gcn",
            "graph/gin",
            "--wandb",
            "--wandb-name",
            "exp",
        ],
    )
    assert [value_of(c, "+logger.wandb.name=") for c in several] == [
        "exp_graph-gcn",
        "exp_graph-gin",
    ]
    single = run_main(
        script,
        monkeypatch,
        ["--model", "graph/gcn", "--wandb", "--wandb-name", "exp"],
    )
    assert value_of(single[0], "+logger.wandb.name=") == "exp"


def test_unknown_model_stops_before_anything_is_trained(script, monkeypatch):
    with pytest.raises(SystemExit, match="graph/nope"):
        run_main(script, monkeypatch, ["--model", "graph/gcn", "graph/nope"])
