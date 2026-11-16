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


def seed_of(cmd):
    return next((a for a in cmd if a.startswith("seed=")), None)


def test_seeds_are_trained_one_after_another_for_each_model(
    script, monkeypatch
):
    commands = run_main(
        script,
        monkeypatch,
        ["--model", "graph/gcn", "graph/gin", "--seed", "0", "1", "2"],
    )
    assert [(value_of(c, "model="), seed_of(c)) for c in commands] == [
        ("graph/gcn", "seed=0"),
        ("graph/gcn", "seed=1"),
        ("graph/gcn", "seed=2"),
        ("graph/gin", "seed=0"),
        ("graph/gin", "seed=1"),
        ("graph/gin", "seed=2"),
    ]


def test_without_seed_the_config_seed_is_used(script, monkeypatch):
    assert seed_of(run_main(script, monkeypatch, [])[0]) is None
    assert seed_of(run_main(script, monkeypatch, ["--seed", "5"])[0]) == (
        "seed=5"
    )


def test_wandb_names_and_tags_carry_the_seed_when_several_are_trained(
    script, monkeypatch
):
    derived = run_main(script, monkeypatch, ["--seed", "1", "2", "--wandb"])
    names = [value_of(c, "+logger.wandb.name=") for c in derived]
    assert names[0].endswith("_seed1") and names[1].endswith("_seed2")
    assert "seed1" in value_of(derived[0], "logger.wandb.tags=")
    explicit = run_main(
        script,
        monkeypatch,
        ["--model", "graph/gcn", "--seed", "1", "2", "--wandb"]
        + ["--wandb-name", "exp"],
    )
    assert [value_of(c, "+logger.wandb.name=") for c in explicit] == [
        "exp_seed1",
        "exp_seed2",
    ]
    single = run_main(
        script, monkeypatch, ["--seed", "1", "--wandb", "--wandb-name", "exp"]
    )
    assert value_of(single[0], "+logger.wandb.name=") == "exp"


def test_a_failing_seed_does_not_stop_the_others(script, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run_main(
            script, monkeypatch, ["--seed", "0", "1", "2"], codes=[0, 4, 0]
        )
    assert stop.value.code == 4


def test_several_seeds_cannot_be_combined_with_a_seed_override(
    script, monkeypatch
):
    with pytest.raises(SystemExit, match="seed"):
        run_main(script, monkeypatch, ["--seed", "0", "1", "--", "seed=3"])
