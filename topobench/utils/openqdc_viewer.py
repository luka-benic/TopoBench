"""Interactive 3D viewer for the edges TopoBench builds from OpenQDC molecules.

OpenQDC stores molecules as point clouds (atomic numbers and positions, no
bonds). TopoBench builds the edges from the geometry, either with the
``bonds`` rule (distance below ``bond_scale`` times the sum of the covalent
radii) or with the ``radius`` rule (distance below ``cutoff``). This module
draws a molecule with both edge sets, which can be switched on and off.

Examples
--------
::

    from topobench.utils.openqdc_viewer import view_molecule, view_molecules

    view_molecule("QM9", index=1234)                        # opens in a browser
    # One page with several molecules and the liftings of some TopoBench models:
    view_molecules("QM9", [3, 17, 1234], "figures", "qm9",
                   models=["simplicial/scn", "cell/cwn", "hypergraph/unignn2"])
    fig = view_molecule("Alchemy", 7, bond_scale=1.2, cutoff=3.0, show=False)
    fig.write_html("molecule.html")

In a Jupyter notebook the figure is shown inline (``fig.show()``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from ase.data import atomic_masses, chemical_symbols
from ase.data.colors import jmol_colors

from topobench.data.utils.openqdc_utils import (
    OpenQDCSource,
    canonical_name,
    default_cache_dir,
    method_names_for,
    molecular_edges,
    resolve_energy_method,
)

EDGE_MODES = ("both", "bonds", "radius", "none")
BOND_COLOR = "#2b2b33"
CUTOFF_COLOR = "#9aa3b2"


def atom_size(numbers: Any, scale: float = 1.0) -> np.ndarray:
    """Marker size (pixels) of each atom, growing with its atomic number.

    The size is proportional to the cube root of the atomic number, so the
    marker volume grows with the number of electrons: hydrogen is 7 pixels,
    carbon about 13 and a heavy metal such as gold about 30. The atomic number
    is used rather than the mass because it separates the elements clearly and
    is the property that identifies an element; the mass is almost monotonic
    in it anyway.

    Parameters
    ----------
    numbers : array-like
        Atomic numbers.
    scale : float
        Overall size factor.

    Returns
    -------
    numpy.ndarray
        Marker sizes in pixels.
    """
    return 7.0 * scale * np.cbrt(np.asarray(numbers, dtype=float))


def _edge_trace(
    positions: np.ndarray,
    edge_index: np.ndarray,
    name: str,
    color: str,
    width: float,
    visible: bool,
) -> go.Scatter3d:
    """Make one line trace holding all edges of a set (one segment each).

    Parameters
    ----------
    positions : numpy.ndarray
        Atom positions, shape ``[N, 3]``.
    edge_index : numpy.ndarray
        Edge index with both directions, shape ``[2, E]``.
    name : str
        Legend name.
    color : str
        Line color.
    width : float
        Line width.
    visible : bool
        Whether the trace is shown at first.

    Returns
    -------
    plotly.graph_objects.Scatter3d
        The line trace (a ``None`` separates the segments).
    """
    keep = edge_index[0] < edge_index[1]  # each undirected edge once
    a, b = edge_index[0][keep], edge_index[1][keep]
    coords: list[list[float | None]] = [[], [], []]
    for i, j in zip(a.tolist(), b.tolist(), strict=True):
        for axis in range(3):
            coords[axis] += [positions[i, axis], positions[j, axis], None]
    return go.Scatter3d(
        x=coords[0],
        y=coords[1],
        z=coords[2],
        mode="lines",
        line={"color": color, "width": width},
        name=f"{name} ({len(a)})",
        legendgroup=name,
        hoverinfo="skip",
        visible=True if visible else "legendonly",
    )


def _molecule_traces(
    numbers: np.ndarray,
    positions: np.ndarray,
    bond_scale: float,
    cutoff: float,
    size_scale: float,
    edges: str,
) -> list[go.BaseTraceType]:
    """Make the traces of one molecule: cutoff edges, bond edges, atoms.

    Parameters
    ----------
    numbers : numpy.ndarray
        Atomic numbers.
    positions : numpy.ndarray
        Positions in Angstrom, shape ``[N, 3]``.
    bond_scale : float
        Factor on the sum of covalent radii for the ``bonds`` rule.
    cutoff : float
        Distance cutoff in Angstrom for the ``radius`` rule.
    size_scale : float
        Factor on the atom marker sizes.
    edges : str
        Edge sets shown at first (one of ``EDGE_MODES``).

    Returns
    -------
    list[plotly.graph_objects.BaseTraceType]
        The cutoff-edge trace, the bond-edge trace, then one trace per element.
    """
    bonds = molecular_edges(
        positions, numbers, "bonds", bond_scale=bond_scale
    ).numpy()
    radius = molecular_edges(
        positions, numbers, "radius", cutoff=cutoff
    ).numpy()

    traces = [
        _edge_trace(
            positions,
            radius,
            f"cutoff edges (< {cutoff:g} Å)",
            CUTOFF_COLOR,
            2,
            edges in ("both", "radius"),
        ),
        _edge_trace(
            positions,
            bonds,
            f"bond edges ({bond_scale:g} × covalent radii)",
            BOND_COLOR,
            6,
            edges in ("both", "bonds"),
        ),
    ]
    for z in sorted(set(numbers.tolist())):
        sel = numbers == z
        idx = np.flatnonzero(sel)
        symbol = chemical_symbols[z]
        traces.append(
            go.Scatter3d(
                x=positions[sel, 0],
                y=positions[sel, 1],
                z=positions[sel, 2],
                mode="markers",
                marker={
                    "size": atom_size(numbers[sel], size_scale),
                    "color": f"rgb{tuple(int(255 * c) for c in jmol_colors[z])}",
                    "line": {"color": "#333333", "width": 1.5},
                    "opacity": 1.0,
                },
                name=f"{symbol} (Z={z}, n={int(sel.sum())})",
                legendgroup=symbol,
                customdata=np.c_[idx, [atomic_masses[z]] * len(idx)],
                hovertemplate=(
                    f"{symbol} #%{{customdata[0]}}<br>Z={z}, "
                    "m=%{customdata[1]:.2f} u<br>"
                    "(%{x:.3f}, %{y:.3f}, %{z:.3f}) Å<extra></extra>"
                ),
            )
        )
    return traces


def view_atoms(
    numbers: Any,
    positions: Any,
    *,
    bond_scale: float = 1.3,
    cutoff: float = 5.0,
    edges: str = "both",
    title: str = "",
    size_scale: float = 1.0,
    show: bool = False,
    html: str | None = None,
) -> go.Figure:
    """Draw one molecule in 3D with its bond-rule and cutoff-rule edges.

    Atoms are colored by element (Jmol colors) and sized by atomic number; the
    legend has one entry per element and one per edge set. Clicking a legend
    entry hides or shows it, and the buttons above the plot switch the edge
    sets.

    Parameters
    ----------
    numbers : array-like
        Atomic numbers, shape ``[N]``.
    positions : array-like
        Positions in Angstrom, shape ``[N, 3]``.
    bond_scale : float
        Factor on the sum of covalent radii for the ``bonds`` rule.
    cutoff : float
        Distance cutoff in Angstrom for the ``radius`` rule.
    edges : str
        Edge sets shown at first: ``"both"``, ``"bonds"``, ``"radius"`` or
        ``"none"``.
    title : str
        Figure title.
    size_scale : float
        Factor on the atom marker sizes.
    show : bool
        Open the figure (browser or notebook) before returning.
    html : str, optional
        Also write the figure to this HTML file.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive figure.
    """
    if edges not in EDGE_MODES:
        raise ValueError(f"edges must be one of {EDGE_MODES}.")
    numbers = np.asarray(numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(numbers), 3):
        raise ValueError("positions must have shape [len(numbers), 3].")

    traces = _molecule_traces(
        numbers, positions, bond_scale, cutoff, size_scale, edges
    )

    def visibility(mode: str) -> list:
        """Visibility list of all traces for an edge mode.

        Parameters
        ----------
        mode : str
            One of ``EDGE_MODES``.

        Returns
        -------
        list
            One entry per trace: the two edge sets first, then the atoms.
        """
        return [
            mode in ("both", "radius"),
            mode in ("both", "bonds"),
        ] + [True] * (len(traces) - 2)

    labels = {
        "both": "Bonds + cutoff",
        "bonds": "Bonds only",
        "radius": "Cutoff only",
        "none": "No edges",
    }
    buttons = [
        {
            "label": labels[m],
            "method": "restyle",
            "args": [{"visible": visibility(m)}],
        }
        for m in EDGE_MODES
    ]
    fig = go.Figure(traces)
    fig.update_layout(
        title={"text": title or "OpenQDC molecule", "x": 0.02},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": buttons,
                "x": 0.02,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "bottom",
                "active": EDGE_MODES.index(edges),
            }
        ],
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x (Å)"},
            "yaxis": {"title": "y (Å)"},
            "zaxis": {"title": "z (Å)"},
        },
        legend={"itemsizing": "constant", "groupclick": "toggleitem"},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        height=700,
    )
    if html:
        fig.write_html(html, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


def view_molecule(
    dataset: str | OpenQDCSource,
    index: int = 0,
    *,
    energy_method: int | str = 0,
    target: str = "energies",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Draw molecule number ``index`` of an OpenQDC dataset.

    The molecule is read with the same loader the training pipelines use (the
    dataset is downloaded first if it is not cached), and drawn with
    :func:`view_atoms`; the title shows its size and energy.

    Parameters
    ----------
    dataset : str or OpenQDCSource
        OpenQDC dataset name (e.g. ``"QM9"``) or an open source.
    index : int
        Index of the molecule in the OpenQDC dataset.
    energy_method : int or str
        Energy column (index or name) shown in the title.
    target : str
        ``"energies"`` (total) or ``"formation_energies"``.
    cache_dir : str, optional
        OpenQDC cache directory; default as in the training scripts.
    **kwargs : Any
        Passed to :func:`view_atoms` (``bond_scale``, ``cutoff``, ``edges``,
        ``size_scale``, ``show``, ``html``).

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive figure.
    """
    if isinstance(dataset, str):
        name = canonical_name(dataset)
        source = OpenQDCSource(name, cache_dir or default_cache_dir())
    else:
        source = dataset
        name = source.name
    column = resolve_energy_method(method_names_for(name), energy_method)
    numbers, _, positions, energy = source.get(index, target, column)
    kwargs.setdefault("show", True)
    kwargs.setdefault(
        "title",
        f"{name} #{index}: {len(numbers)} atoms, {source.method_names[column]}"
        f" {target} = {energy:.4f} eV",
    )
    return view_atoms(numbers, positions, **kwargs)


LIFT_COLOR = "lightpink"

GALLERY_JS = """
(function () {
  var gd = document.getElementById('{plot_id}');
  var cfg = __CONFIG__;
  var st = {mol: 0, bonds: cfg.bonds, cutoff: cfg.cutoff, lift: -1, bs: 0, cs: 0, graph: 'bonds', method: cfg.method};
  var bar = document.createElement('div');
  bar.style.cssText = 'font:14px sans-serif;padding:8px 12px;display:flex;' +
    'flex-wrap:wrap;gap:14px;align-items:center;';
  function el(tag, props) {
    var e = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) { e[k] = props[k]; });
    return e;
  }
  function label(text, child) {
    var l = el('label'); l.style.cssText = 'display:flex;gap:5px;align-items:center;';
    l.appendChild(child); l.appendChild(document.createTextNode(text)); return l;
  }
  var prev = el('button', {textContent: '\\u25C0'});
  var next = el('button', {textContent: '\\u25B6'});
  var molSel = el('select');
  cfg.titles.forEach(function (t, i) { molSel.appendChild(el('option', {value: i, textContent: t})); });
  var bonds = el('input', {type: 'checkbox', checked: cfg.bonds});
  var cut = el('input', {type: 'checkbox', checked: cfg.cutoff});
  var liftSel = el('select');
  liftSel.appendChild(el('option', {value: -1, textContent: 'no lifting'}));
  cfg.liftings.forEach(function (t, i) { liftSel.appendChild(el('option', {value: i, textContent: t})); });
  var bsSel = el('select');
  cfg.bondScales.forEach(function (v, i) { bsSel.appendChild(el('option', {value: i, textContent: v})); });
  var csSel = el('select');
  cfg.cutoffs.forEach(function (v, i) { csSel.appendChild(el('option', {value: i, textContent: v})); });
  var methSel = el('select');
  cfg.methods.forEach(function (n, i) { methSel.appendChild(el('option', {value: i, textContent: i + ': ' + n})); });
  methSel.value = cfg.method;
  var graphSel = el('select');
  graphSel.appendChild(el('option', {value: 'bonds', textContent: 'bond edges'}));
  graphSel.appendChild(el('option', {value: 'radius', textContent: 'cutoff edges'}));
  function field(text, child) {
    var l = el('label'); l.style.cssText = 'display:flex;gap:5px;align-items:center;';
    l.appendChild(document.createTextNode(text)); l.appendChild(child); return l;
  }
  var molBox = el('span'); molBox.style.cssText = 'display:flex;gap:4px;align-items:center;';
  molBox.appendChild(prev); molBox.appendChild(molSel); molBox.appendChild(next);
  bar.appendChild(molBox);
  bar.appendChild(label('bond edges', bonds));
  bar.appendChild(label('cutoff edges', cut));
  if (cfg.methods.length > 1) bar.appendChild(field('energy method', methSel));
  if (cfg.bondScales.length > 1) bar.appendChild(field('bond scale', bsSel));
  if (cfg.cutoffs.length > 1) bar.appendChild(field('cutoff (\u00c5)', csSel));
  if (cfg.liftings.length) {
    bar.appendChild(field('lifting', liftSel));
    bar.appendChild(field('lifted graph', graphSel));
  }
  gd.parentNode.insertBefore(bar, gd);

  function title() {
    var e = cfg.energies[st.mol][st.method];
    return cfg.titles[st.mol] + ', ' + cfg.methods[st.method] + ' ' + cfg.target + ' = ' +
      (e === null ? 'not computed' : e.toFixed(4) + ' eV');
  }
  function update() {
    var vis = gd.data.map(function (t) {
      var m = t.meta || {};
      if (m.mol !== st.mol) return false;
      if (m.kind === 'bonds') return st.bonds && m.v === st.bs;
      if (m.kind === 'cutoff') return st.cutoff && m.v === st.cs;
      if (m.kind === 'lifting')
        return m.lift === st.lift && m.graph === st.graph &&
          m.v === (st.graph === 'bonds' ? st.bs : st.cs);
      return true;
    });
    Plotly.restyle(gd, {visible: vis});
    Plotly.relayout(gd, {'title.text': title()});
    molSel.value = st.mol;
  }
  function go(d) { st.mol = (st.mol + d + cfg.titles.length) % cfg.titles.length; update(); }
  prev.onclick = function () { go(-1); };
  next.onclick = function () { go(1); };
  molSel.onchange = function () { st.mol = +molSel.value; update(); };
  bonds.onchange = function () { st.bonds = bonds.checked; update(); };
  cut.onchange = function () { st.cutoff = cut.checked; update(); };
  liftSel.onchange = function () { st.lift = +liftSel.value; update(); };
  bsSel.onchange = function () { st.bs = +bsSel.value; update(); };
  csSel.onchange = function () { st.cs = +csSel.value; update(); };
  graphSel.onchange = function () { st.graph = graphSel.value; update(); };
  methSel.onchange = function () { st.method = +methSel.value; update(); };
  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') go(-1);
    if (e.key === 'ArrowRight') go(1);
  });
  update();
})();
"""


def _plane_order(points: np.ndarray) -> np.ndarray:
    """Order points of a roughly planar cell by angle around their centroid.

    Parameters
    ----------
    points : numpy.ndarray
        Points of one cell, shape ``[n, 3]``.

    Returns
    -------
    numpy.ndarray
        Indices of the points in angular order in their best-fit plane.
    """
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u, v = centered @ vt[0], centered @ vt[1]
    return np.argsort(np.arctan2(v, u))


def _cell_faces(
    points: np.ndarray, rank2: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the surface of one cell for drawing.

    Parameters
    ----------
    points : numpy.ndarray
        Points of the cell, shape ``[n, 3]`` with ``n >= 3``.
    rank2 : bool
        True for a polygon (a 2-cell): it is filled as a fan around its
        centroid. Otherwise the convex hull is used (3-cells, hyperedges),
        falling back to the fan when the points are coplanar.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The vertices (the points, plus the centroid for a fan) and the
        triangles as vertex indices, shape ``[T, 3]``.
    """
    if len(points) == 3:
        return points, np.array([[0, 1, 2]])
    if not rank2:
        try:
            from scipy.spatial import ConvexHull, QhullError

            return points, ConvexHull(points).simplices
        except QhullError:
            pass
    order = _plane_order(points)
    vertices = np.vstack([points[order], points.mean(axis=0)])
    n = len(points)
    triangles = [[n, k, (k + 1) % n] for k in range(n)]
    return vertices, np.array(triangles)


def _lifting_traces(
    positions: np.ndarray,
    cells: dict[str, list[tuple[int, ...]]],
    label: str,
    meta: dict[str, Any],
    visible: bool,
    max_cells: int | None = None,
) -> list[go.BaseTraceType]:
    """Make the light-pink traces that draw the cells of a lifted molecule.

    Parameters
    ----------
    positions : numpy.ndarray
        Atom positions, shape ``[N, 3]``.
    cells : dict[str, list[tuple[int, ...]]]
        Cells per rank, as returned by ``lifted_cells``.
    label : str
        Name of the lifting, used in the legend.
    meta : dict[str, Any]
        Metadata stored on every trace (read by the page controls).
    visible : bool
        Whether the traces are shown at first.
    max_cells : int, optional
        Draw at most this many cells per rank (the first ones); the legend
        says how many of how many are shown.

    Returns
    -------
    list[plotly.graph_objects.BaseTraceType]
        One filled-surface trace per rank (cells with 3 or more atoms) and one
        line trace for cells with 2 atoms.
    """
    traces: list[go.BaseTraceType] = []
    for rank, all_members in cells.items():
        members = all_members[:max_cells] if max_cells else all_members
        shown = (
            f"first {len(members)} of {len(all_members)}"
            if len(members) < len(all_members)
            else None
        )
        xs, ys, zs, ti, tj, tk, text = [], [], [], [], [], [], []
        lx, ly, lz = [], [], []
        n_surfaces = n_segments = 0
        for number, atoms in enumerate(members):
            if len(atoms) < 2:
                continue
            points = positions[list(atoms)]
            if len(atoms) == 2:
                for axis, out in enumerate((lx, ly, lz)):
                    out += [points[0, axis], points[1, axis], None]
                n_segments += 1
                continue
            vertices, triangles = _cell_faces(points, rank == "rank 2")
            offset = len(xs)
            xs += vertices[:, 0].tolist()
            ys += vertices[:, 1].tolist()
            zs += vertices[:, 2].tolist()
            ti += (triangles[:, 0] + offset).tolist()
            tj += (triangles[:, 1] + offset).tolist()
            tk += (triangles[:, 2] + offset).tolist()
            text += [f"{rank} #{number}: atoms {list(atoms)}"] * len(vertices)
            n_surfaces += 1
        if n_surfaces:
            traces.append(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ti,
                    j=tj,
                    k=tk,
                    color=LIFT_COLOR,
                    opacity=0.6,
                    flatshading=True,
                    name=f"{label}: {rank} ({shown or n_surfaces})",
                    legendgroup=f"{label}-{rank}",
                    showlegend=True,
                    hovertext=text,
                    hoverinfo="text",
                    meta=meta,
                    visible=visible,
                )
            )
        if n_segments:
            traces.append(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    line={"color": LIFT_COLOR, "width": 8},
                    name=f"{label}: {rank} of 2 atoms ({shown or n_segments})",
                    legendgroup=f"{label}-{rank}-pairs",
                    hoverinfo="skip",
                    meta=meta,
                    visible=visible,
                )
            )
    if not traces:  # keep an entry in the legend: nothing found is a result
        traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"color": LIFT_COLOR, "symbol": "x"},
                name=f"{label}: no cells of rank 2 or higher on this graph",
                hoverinfo="skip",
                meta=meta,
                visible=visible,
            )
        )
    return traces


def resolve_molecules(
    dataset: str | OpenQDCSource,
    indices: Sequence[int],
    base_name: str,
    cache_dir: str | None = None,
) -> tuple[OpenQDCSource, list[int]]:
    """Open a dataset and check the molecule indices and the file name.

    Parameters
    ----------
    dataset : str or OpenQDCSource
        OpenQDC dataset name or an open source.
    indices : sequence of int
        Indices of the molecules to show.
    base_name : str
        File name without suffix; must not contain a path.
    cache_dir : str, optional
        OpenQDC cache directory; default as in the training scripts.

    Returns
    -------
    tuple[OpenQDCSource, list[int]]
        The source and the indices as a list of ints.

    Raises
    ------
    ValueError
        If the indices are empty or repeated, or ``base_name`` is a path.
    IndexError
        If an index is outside the dataset.
    """
    indices = [int(i) for i in indices]
    if not indices:
        raise ValueError("indices is empty.")
    if len(set(indices)) != len(indices):
        raise ValueError("indices contains duplicates.")
    if not base_name or Path(base_name).name != base_name:
        raise ValueError("base_name must be a plain file name, not a path.")
    if isinstance(dataset, str):
        source = OpenQDCSource(
            canonical_name(dataset), cache_dir or default_cache_dir()
        )
    else:
        source = dataset
    bad = [i for i in indices if not 0 <= i < len(source)]
    if bad:
        raise IndexError(
            f"Indices {bad} are outside 0..{len(source) - 1} of {source.name}."
        )
    return source, indices


def _as_values(value: float | Sequence[float], name: str) -> list[float]:
    """Turn a number or a list of numbers into a list of distinct floats.

    Parameters
    ----------
    value : float or sequence of float
        One value or several.
    name : str
        Parameter name, for the error message.

    Returns
    -------
    list[float]
        The values in the given order.
    """
    values = (
        [float(value)] if np.isscalar(value) else [float(v) for v in value]
    )
    if not values or len(set(values)) != len(values) or min(values) <= 0:
        raise ValueError(f"{name} needs distinct positive values.")
    return values


def view_molecules(
    dataset: str | OpenQDCSource,
    indices: Sequence[int],
    save_dir: str | Path,
    base_name: str,
    *,
    models: Sequence[str] = (),
    energy_method: int | str = 0,
    target: str = "energies",
    cache_dir: str | None = None,
    config_dir: str | Path | None = None,
    bond_scale: float | Sequence[float] = 1.3,
    cutoff: float | Sequence[float] = 5.0,
    edges: str = "both",
    size_scale: float = 1.0,
    max_cells: int | None = 3000,
    offline: bool = False,
) -> Path:
    """Save several molecules of an OpenQDC dataset in one interactive page.

    The page ``<save_dir>/<base_name>.html`` shows one molecule at a time. Above
    the plot you can switch between the molecules (menu, arrows or the left
    and right keys), switch the energy method shown in the title, turn the bond edges and the cutoff edges on and off, and,
    if ``models`` is given, switch between the liftings those TopoBench models
    apply to the molecule. A lifting draws its higher-order cells (simplices,
    cells or hyperedges) in light pink.

    The liftings follow the graph: every lifting is computed on the graph the
    bond rule gives and on the graph the cutoff rule gives, and the *lifted
    graph* menu chooses which one is shown. ``bond_scale`` and ``cutoff`` may
    be lists; the page then has a menu for each, and edges and liftings are
    shown for the chosen value.

    The dataset is opened once, and every index is checked before anything is
    written.

    Parameters
    ----------
    dataset : str or OpenQDCSource
        OpenQDC dataset name (e.g. ``"QM9"``) or an open source.
    indices : sequence of int
        Indices of the molecules in the OpenQDC dataset.
    save_dir : str or Path
        Folder for the HTML file (created if missing).
    base_name : str
        File name without the ``.html`` suffix.
    models : sequence of str
        TopoBench model names such as ``"simplicial/scn"``, ``"cell/cwn"`` or
        ``"hypergraph/unignn2"``. Their liftings are resolved from the Hydra
        configs as in a training run. Models that run on the plain graph add
        nothing, and models that share a lifting share one menu entry.
    energy_method : int or str
        Energy column (index or name) shown at first. The page also holds the
        energy of every geometry for all the other methods, and a menu
        switches between them.
    target : str
        ``"energies"`` (total) or ``"formation_energies"``.
    cache_dir : str, optional
        OpenQDC cache directory; default as in the training scripts.
    config_dir : str or Path, optional
        The TopoBench ``configs`` folder; default: the one of this repository.
    bond_scale : float or sequence of float
        Factor(s) on the sum of covalent radii for the bond edges; the first
        value is shown at first.
    cutoff : float or sequence of float
        Distance cutoff(s) in Angstrom for the cutoff edges; the first value is
        shown at first.
    edges : str
        Edge sets shown at first: ``"both"``, ``"bonds"``, ``"radius"`` or
        ``"none"``.
    size_scale : float
        Factor on the atom marker sizes.
    max_cells : int, optional
        Draw at most this many cells per rank and lifting (dense cutoff graphs
        can have thousands); the legend says how many are shown. None draws
        all.
    offline : bool
        Embed the Plotly library in the page (about 5 MB) so that it opens
        without internet access; by default it is loaded from a CDN.

    Returns
    -------
    pathlib.Path
        The written HTML file.
    """
    import json

    bond_scales = _as_values(bond_scale, "bond_scale")
    cutoffs = _as_values(cutoff, "cutoff")
    if edges not in EDGE_MODES:
        raise ValueError(f"edges must be one of {EDGE_MODES}.")
    source, indices = resolve_molecules(dataset, indices, base_name, cache_dir)

    # One menu entry per distinct lifting; models that share one are listed
    # together.
    groups: dict[str, dict[str, Any]] = {}
    if models:
        from topobench.utils.openqdc_liftings import (
            lifting_key,
            lifting_label,
            model_liftings,
        )

        for model in models:
            liftings = model_liftings(model, config_dir)
            if not liftings:
                print(f"[viewer] {model} runs on the plain graph: no lifting.")
                continue
            group = groups.setdefault(
                lifting_key(liftings),
                {
                    "liftings": liftings,
                    "models": [],
                    "name": lifting_label(liftings),
                },
            )
            group["models"].append(model)
    lift_to_cells = None
    if groups:
        from topobench.utils.openqdc_liftings import lift_to_cells
    group_list = list(groups.values())
    for group in group_list:
        group["label"] = f"{', '.join(group['models'])}: {group['name']}"

    column = resolve_energy_method(
        method_names_for(source.name), energy_method
    )
    n_methods = len(source.method_names)
    show_bonds = edges in ("both", "bonds")
    show_cutoff = edges in ("both", "radius")
    traces: list[go.BaseTraceType] = []
    titles = []
    energies = []
    failures = []
    for position, index in enumerate(indices):
        numbers, _, coords, _ = source.get(index, target, column)
        numbers = np.asarray(numbers, dtype=int)
        coords = np.asarray(coords, dtype=float)
        titles.append(f"{source.name} #{index}: {len(numbers)} atoms")
        # The energy of this geometry for every method; the page switches
        # between them.
        values = [source.get(index, target, c)[3] for c in range(n_methods)]
        energies.append([float(v) if np.isfinite(v) else None for v in values])
        first = position == 0
        for trace in _molecule_traces(
            numbers, coords, bond_scales[0], cutoffs[0], size_scale, "none"
        )[2:]:  # the atoms
            trace.meta = {"mol": position, "kind": "atoms"}
            trace.visible = first
            traces.append(trace)
        for kind, rule, values, color, width, name, shown in (
            (
                "bonds",
                "bonds",
                bond_scales,
                BOND_COLOR,
                6,
                "bond edges ({:g} × covalent radii)",
                show_bonds,
            ),
            (
                "cutoff",
                "radius",
                cutoffs,
                CUTOFF_COLOR,
                2,
                "cutoff edges (< {:g} Å)",
                show_cutoff,
            ),
        ):
            for v, value in enumerate(values):
                edge_index = molecular_edges(
                    coords,
                    numbers,
                    rule,
                    bond_scale=value if rule == "bonds" else bond_scales[0],
                    cutoff=value if rule == "radius" else cutoffs[0],
                ).numpy()
                trace = _edge_trace(
                    coords,
                    edge_index,
                    name.format(value),
                    color,
                    width,
                    True,
                )
                trace.meta = {"mol": position, "kind": kind, "v": v}
                trace.visible = bool(first and shown and v == 0)
                traces.append(trace)
        for number, group in enumerate(group_list):
            for rule, values in (("bonds", bond_scales), ("radius", cutoffs)):
                for v, value in enumerate(values):
                    try:
                        cells = lift_to_cells(
                            numbers,
                            coords,
                            group["liftings"],
                            graph_edges=rule,
                            bond_scale=value
                            if rule == "bonds"
                            else bond_scales[0],
                            cutoff=value if rule == "radius" else cutoffs[0],
                        )
                    except Exception as error:  # odd molecules
                        failures.append(
                            f"#{index} {group['name']} on the {rule} graph "
                            f"({value:g}): {type(error).__name__}: {error}"
                        )
                        continue
                    meta = {
                        "mol": position,
                        "kind": "lifting",
                        "lift": number,
                        "graph": rule,
                        "v": v,
                    }
                    traces += _lifting_traces(
                        coords, cells, group["name"], meta, False, max_cells
                    )
    for failure in failures:
        print(f"[viewer] lifting failed for {failure}")

    fig = go.Figure(traces)
    fig.update_layout(
        title={"text": "", "x": 0.02},
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x (Å)"},
            "yaxis": {"title": "y (Å)"},
            "zaxis": {"title": "z (Å)"},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        height=720,
    )
    config = {
        "titles": titles,
        "methods": list(source.method_names),
        "method": column,
        "energies": energies,
        "target": target,
        "liftings": [g["label"] for g in group_list],
        "bonds": show_bonds,
        "cutoff": show_cutoff,
        "bondScales": [f"{v:g}" for v in bond_scales],
        "cutoffs": [f"{v:g}" for v in cutoffs],
    }
    script = GALLERY_JS.replace("__CONFIG__", json.dumps(config))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{base_name}.html"
    path.write_text(
        fig.to_html(
            include_plotlyjs=True if offline else "cdn", post_script=script
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Command line entry: write several molecules to one HTML page."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="e.g. QM9")
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0],
        help="molecule indices shown in the page",
    )
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--base-name", default="molecule")
    parser.add_argument("--energy-method", default=0)
    parser.add_argument(
        "--bond-scale",
        type=float,
        nargs="+",
        default=[1.3],
        help="factor(s) on the sum of covalent radii (bond edges); several "
        "values give a menu in the page",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        nargs="+",
        default=[5.0],
        help="distance cutoff(s) in Angstrom (cutoff edges); several values "
        "give a menu in the page",
    )
    parser.add_argument("--edges", default="both", choices=EDGE_MODES)
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="TopoBench models whose liftings are drawn, e.g. simplicial/scn "
        "cell/cwn hypergraph/unignn2",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=3000,
        help="draw at most this many cells per rank and lifting",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="embed the Plotly library so the page opens without internet",
    )
    args = parser.parse_args()
    energy_method = args.energy_method
    if isinstance(energy_method, str) and energy_method.isdigit():
        energy_method = int(energy_method)
    path = view_molecules(
        args.dataset,
        args.indices,
        args.save_dir,
        args.base_name,
        models=args.models,
        max_cells=args.max_cells,
        energy_method=energy_method,
        bond_scale=args.bond_scale,
        cutoff=args.cutoff,
        edges=args.edges,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    print(f"Wrote {path}; open it in a browser.")


if __name__ == "__main__":
    main()


__all__ = ["atom_size", "view_atoms", "view_molecule", "view_molecules"]
