"""Save test-set predictions of a model as ``.txt`` files and a parity plot."""

import os
from pathlib import Path
from typing import Any

import numpy as np

# Chart colors (light surface), taken from the project's reference palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
# One-hue sequential ramp (blue steps 100 to 700) for the density plot.
SEQUENTIAL = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
# From this many points on, a density plot replaces the scatter plot.
DENSITY_FROM = 3000


def regression_metrics(
    reference: Any, prediction: Any, n_atoms: Any = None
) -> dict[str, float]:
    """Compute error metrics of predicted against reference energies.

    Parameters
    ----------
    reference : array-like
        Reference (ground-truth) energies in eV.
    prediction : array-like
        Predicted energies in eV.
    n_atoms : array-like, optional
        Number of atoms of each molecule; adds a per-atom error.

    Returns
    -------
    dict[str, float]
        Metrics: number of molecules, MAE, RMSE, R2, bias (mean of
        prediction minus reference), median, 95th percentile and maximum of
        the absolute error, and the per-atom MAE if ``n_atoms`` is given.
    """
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    error = pred - ref
    absolute = np.abs(error)
    total = float(np.sum((ref - ref.mean()) ** 2))
    metrics = {
        "n": float(len(ref)),
        "mae_eV": float(absolute.mean()),
        "mae_meV": float(1000.0 * absolute.mean()),
        "rmse_eV": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / total) if total > 0 else np.nan,
        "bias_eV": float(error.mean()),
        "median_abs_error_eV": float(np.median(absolute)),
        "p95_abs_error_eV": float(np.percentile(absolute, 95)),
        "max_abs_error_eV": float(absolute.max()),
    }
    if n_atoms is not None:
        per_atom = absolute / np.asarray(n_atoms, dtype=float)
        metrics["mae_per_atom_meV"] = float(1000.0 * per_atom.mean())
    return metrics


def write_predictions_txt(
    path: str | Path, columns: dict[str, Any], header: list[str]
) -> None:
    """Write per-molecule columns to a whitespace-separated ``.txt`` file.

    The header lines start with ``#``, so the file loads with
    ``numpy.loadtxt``.

    Parameters
    ----------
    path : str or Path
        Output file.
    columns : dict[str, array-like]
        Column name to values; all columns have the same length.
    header : list[str]
        Comment lines written at the top of the file.
    """
    names = list(columns)
    text_columns = []
    for name in names:
        values = np.asarray(columns[name])
        if np.issubdtype(values.dtype, np.integer):
            text_columns.append(values.astype(str))
        else:
            text_columns.append(np.char.mod("%.6f", values.astype(float)))
    lines = [f"# {line}" for line in header]
    lines.append("# columns: " + " ".join(names))
    lines += [" ".join(row) for row in zip(*text_columns, strict=True)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_split(
    directory: str | Path, split_ids: dict[str, Any], header: list[str]
) -> Path:
    """Write the molecules of each split of a run to ``split.txt``.

    The file has one line per molecule, ``openqdc_idx`` and the name of its
    split (``train``, ``valid`` or ``test``), and loads with
    ``numpy.loadtxt(path, dtype=str)``.

    Parameters
    ----------
    directory : str or Path
        Folder that receives ``split.txt`` (created if missing).
    split_ids : dict[str, array-like]
        Split name to the OpenQDC indices of its molecules.
    header : list[str]
        Comment lines written at the top of the file.

    Returns
    -------
    Path
        The written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [f"# {line}" for line in header]
    lines.append(
        "# counts: "
        + " ".join(f"{name}={len(ids)}" for name, ids in split_ids.items())
    )
    lines.append("# columns: openqdc_idx split")
    for name, ids in split_ids.items():
        lines += [f"{int(i)} {name}" for i in ids]
    path = directory / "split.txt"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_metrics_txt(
    path: str | Path, metrics: dict[str, float], header: list[str]
) -> None:
    """Write metrics to a ``.txt`` file with one ``name value`` pair per line.

    Parameters
    ----------
    path : str or Path
        Output file.
    metrics : dict[str, float]
        Metric name to value.
    header : list[str]
        Comment lines written at the top of the file.
    """
    lines = [f"# {line}" for line in header]
    lines.append("# columns: metric value")
    for name, value in metrics.items():
        text = f"{int(value)}" if name == "n" else f"{value:.6g}"
        lines.append(f"{name} {text}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parity_plot(
    path: str | Path,
    reference: Any,
    prediction: Any,
    title: str,
    metrics: dict[str, float],
    quantity: str = "Energy",
) -> None:
    """Save a parity plot of predicted against reference energies.

    Up to ``DENSITY_FROM`` molecules are drawn as points. More molecules are
    drawn as a density plot (counts per bin on a log scale), so that the
    points do not cover each other.

    Parameters
    ----------
    path : str or Path
        Output image file (PNG).
    reference : array-like
        Reference (ground-truth) energies in eV.
    prediction : array-like
        Predicted energies in eV.
    title : str
        Title of the plot.
    metrics : dict[str, float]
        Metrics from :func:`regression_metrics`, shown in the plot.
    quantity : str
        Name of the plotted quantity used in the axis labels.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    low = float(min(ref.min(), pred.min()))
    high = float(max(ref.max(), pred.max()))
    pad = 0.04 * (high - low if high > low else 1.0)
    limits = (low - pad, high + pad)

    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.plot(limits, limits, color=MUTED, linewidth=1.2, zorder=2)

    if len(ref) < DENSITY_FROM:
        ax.scatter(
            ref,
            pred,
            s=16,
            color=SERIES,
            alpha=0.65,
            edgecolors=SURFACE,
            linewidths=0.4,
            zorder=3,
        )
    else:
        cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
        bins = ax.hexbin(
            ref,
            pred,
            gridsize=70,
            extent=(*limits, *limits),
            mincnt=1,
            bins="log",
            cmap=cmap,
            linewidths=0.0,
            zorder=3,
        )
        colorbar = fig.colorbar(bins, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(
            "molecules per bin (log scale)", color=INK_SECONDARY, fontsize=9
        )
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)

    lines = [f"N = {int(metrics['n'])}", f"MAE = {metrics['mae_eV']:.4g} eV"]
    lines.append(f"RMSE = {metrics['rmse_eV']:.4g} eV")
    lines.append(f"R² = {metrics['r2']:.5f}")
    ax.text(
        0.04,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=INK_SECONDARY,
        linespacing=1.5,
    )
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel(f"Reference {quantity.lower()} (eV)", color=INK_SECONDARY)
    ax.set_ylabel(f"Predicted {quantity.lower()} (eV)", color=INK_SECONDARY)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def log_results_to_wandb(
    run: Any, saved: dict[str, Any], prefix: str = "test_results"
) -> None:
    """Log saved test-set results to a Weights & Biases run.

    The parity plot is logged as an image, the metrics are added to the run
    summary, and the ``.txt`` files (predictions, metrics and the run's ``split.txt``) are
    uploaded with the run.

    Parameters
    ----------
    run : Any
        The wandb run (``wandb.init(...)`` or ``WandbLogger.experiment``).
    saved : dict[str, Any]
        The result of :func:`save_test_results`.
    prefix : str
        Prefix of the logged names.
    """
    import wandb

    run.log({f"{prefix}/parity": wandb.Image(saved["parity_file"])})
    run.summary.update(
        {f"{prefix}/{name}": value for name, value in saved["metrics"].items()}
    )
    folder = Path(saved["predictions_file"]).parent
    files = [saved["predictions_file"], saved["metrics_file"]]
    if (folder / "split.txt").exists():  # written by the run before training
        files.append(folder / "split.txt")
    for file in files:
        run.save(str(file), base_path=str(folder), policy="now")


def save_test_results(
    output_dir: str | Path,
    reference: Any,
    prediction: Any,
    n_atoms: Any = None,
    index: Any = None,
    info: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write test-set predictions, metrics and a parity plot to a folder.

    Three files are written: ``test_predictions.txt`` (one row per molecule),
    ``test_metrics.txt`` and ``test_parity.png``.

    Parameters
    ----------
    output_dir : str or Path
        Folder for the files; it is created if needed.
    reference : array-like
        Reference (ground-truth) energies in eV.
    prediction : array-like
        Predicted energies in eV.
    n_atoms : array-like, optional
        Number of atoms of each molecule.
    index : array-like, optional
        OpenQDC molecule index of each molecule.
    info : dict[str, str], optional
        Description written into the file headers and the plot title. Known
        keys: ``model``, ``dataset``, ``energy_column``, ``quantity`` and
        ``note``.

    Returns
    -------
    dict[str, Any]
        The paths of the three files (``predictions_file``,
        ``metrics_file``, ``parity_file``) and the computed ``metrics``.
    """
    info = info or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)

    header = ["OpenQDC test-set results"]
    for key, label in [
        ("model", "model"),
        ("dataset", "dataset"),
        ("energy_column", "energy column"),
        ("quantity", "quantity"),
        ("note", "note"),
    ]:
        if info.get(key):
            header.append(f"{label}: {info[key]}")
    header.append("units: eV; error = prediction - reference")

    columns: dict[str, Any] = {}
    if index is not None:
        columns["openqdc_idx"] = np.asarray(index, dtype=np.int64)
    if n_atoms is not None:
        columns["n_atoms"] = np.asarray(n_atoms, dtype=np.int64)
    columns["reference_eV"] = ref
    columns["prediction_eV"] = pred
    columns["error_eV"] = pred - ref

    metrics = regression_metrics(ref, pred, n_atoms)
    paths = {
        "predictions": out / "test_predictions.txt",
        "metrics": out / "test_metrics.txt",
        "parity": out / "test_parity.png",
    }
    write_predictions_txt(paths["predictions"], columns, header)
    write_metrics_txt(paths["metrics"], metrics, header)
    title = (
        f"{info.get('model', 'model')} on {info.get('dataset', '')}: test set"
    )
    parity_plot(
        paths["parity"],
        ref,
        pred,
        title.strip(),
        metrics,
        info.get("quantity", "energy"),
    )
    return {
        "predictions_file": str(paths["predictions"]),
        "metrics_file": str(paths["metrics"]),
        "parity_file": str(paths["parity"]),
        "metrics": metrics,
    }
