"""Tests for saving test-set predictions as .txt files and parity plots."""

from pathlib import Path

import numpy as np
import pytest

from topobench.utils.openqdc_results import (
    DENSITY_FROM,
    regression_metrics,
    save_test_results,
)


def test_regression_metrics_values():
    """Metrics of a known error pattern are computed correctly."""
    reference = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = reference + np.array([0.1, -0.1, 0.2, -0.2])
    metrics = regression_metrics(
        reference, prediction, n_atoms=np.array([1, 2, 1, 2])
    )
    assert metrics["n"] == 4
    assert metrics["mae_eV"] == pytest.approx(0.15)
    assert metrics["mae_meV"] == pytest.approx(150.0)
    assert metrics["rmse_eV"] == pytest.approx(np.sqrt(0.025))
    assert metrics["bias_eV"] == pytest.approx(0.0)
    assert metrics["max_abs_error_eV"] == pytest.approx(0.2)
    assert metrics["mae_per_atom_meV"] == pytest.approx(
        1000 * np.mean([0.1, 0.05, 0.2, 0.1])
    )
    assert metrics["r2"] == pytest.approx(1 - 0.1 / 5.0)


@pytest.mark.parametrize("n", [50, DENSITY_FROM + 10])
def test_save_test_results_writes_all_files(tmp_path, n):
    """The .txt files load with numpy and the plot is written (both plot types)."""
    rng = np.random.default_rng(0)
    reference = rng.normal(-90.0, 10.0, n)
    prediction = reference + rng.normal(0.0, 0.1, n)
    saved = save_test_results(
        tmp_path / "out",
        reference,
        prediction,
        n_atoms=rng.integers(10, 30, n),
        index=np.arange(n) + 1000,
        info={"model": "graph/gcn", "dataset": "Alchemy"},
    )
    table = np.loadtxt(saved["predictions_file"])
    assert table.shape == (n, 5)
    assert table[0, 0] == 1000  # openqdc_idx
    assert np.allclose(table[:, 4], table[:, 3] - table[:, 2], atol=2e-6)
    lines = Path(saved["predictions_file"]).read_text(encoding="utf-8")
    assert lines.startswith("# ")
    metrics = {
        line.split()[0]: float(line.split()[1])
        for line in Path(saved["metrics_file"]).read_text().splitlines()
        if not line.startswith("#")
    }
    assert metrics["n"] == n
    assert metrics["mae_eV"] == pytest.approx(
        saved["metrics"]["mae_eV"], rel=1e-5
    )
    assert Path(saved["parity_file"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_test_results_without_optional_columns(tmp_path):
    """Without molecule index and atom counts only the energy columns remain."""
    reference = np.linspace(-10.0, -5.0, 20)
    saved = save_test_results(tmp_path, reference, reference + 0.01)
    table = np.loadtxt(saved["predictions_file"])
    assert table.shape == (20, 3)
    assert "mae_per_atom_meV" not in saved["metrics"]


def test_log_results_to_wandb_uses_run_api(tmp_path):
    """The parity image, metrics and both .txt files go to the wandb run."""
    pytest.importorskip("wandb")
    from topobench.utils.openqdc_results import log_results_to_wandb

    class FakeRun:
        def __init__(self):
            self.logged, self.summary_updates, self.saved = [], {}, []
            self.summary = self

        def log(self, data):
            self.logged.append(data)

        def update(self, values):
            self.summary_updates.update(values)

        def save(self, path, base_path=None, policy=None):
            self.saved.append(Path(path).name)

    reference = np.linspace(-10.0, -5.0, 30)
    saved = save_test_results(tmp_path, reference, reference + 0.05)
    run = FakeRun()
    log_results_to_wandb(run, saved)
    assert "test_results/parity" in run.logged[0]
    assert run.summary_updates["test_results/mae_eV"] == pytest.approx(0.05)
    assert sorted(run.saved) == ["test_metrics.txt", "test_predictions.txt"]


def test_save_split_writes_one_line_per_molecule(tmp_path):
    from topobench.utils.openqdc_results import save_split

    path = save_split(
        tmp_path / "run",
        {"train": [5, 1], "valid": [7], "test": [3]},
        ["a run"],
    )
    rows = [
        line.split()
        for line in path.read_text().splitlines()
        if not line.startswith("#")
    ]
    assert rows == [
        ["5", "train"],
        ["1", "train"],
        ["7", "valid"],
        ["3", "test"],
    ]
