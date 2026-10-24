"""Interactive view of the graph MACE builds: cutoff radius and number of layers.

MACE connects every pair of atoms closer than ``r_max`` and passes messages
along these edges once per interaction layer. After ``L`` layers the energy of
an atom therefore depends on all atoms within ``L`` hops of that graph, never
farther than ``L * r_max`` in distance, and each edge counts less and less
towards the cutoff because of the smooth polynomial envelope.

This module shows that for real molecules, from the geometry alone: no model
is built, run or trained. For a chosen centre atom it draws

* the atoms the centre's energy can depend on (its *receptive field*): the
  atoms within ``L`` hops, coloured by hop;
* the radius graph, with the edges faded by MACE's cutoff envelope;
* the spheres of radius ``r_max`` and ``L * r_max``;
* a plot of the hop count of every atom against its distance from the centre.

Examples
--------
::

    from topobench.utils.mace_viewer import view_mace_receptive_field

    view_mace_receptive_field("QM9", [3, 17], "figures", "qm9_mace",
                              r_max=[2, 3, 4, 5, 6], layers=[1, 2, 3, 4])
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from ase.data import chemical_symbols
from plotly.subplots import make_subplots

from topobench.utils.openqdc_viewer import atom_size, resolve_molecules

POLYNOMIAL_CUTOFF = 5  # MACE's default exponent of the cutoff envelope
HOP_COLORS = ["#1b6ca8", "#2a9d8f", "#e9a23b", "#e76f51", "#8e5ea2", "#6d4c41"]
OUTSIDE_COLOR = "#c5c9d1"
CENTER_COLOR = "#111111"

PAGE_JS = """
(function () {
  var gd = document.getElementById('{plot_id}');
  var cfg = __CONFIG__;
  var st = {mol: 0, c: 0, r: cfg.start.r, l: cfg.start.l,
            edges: true, s1: true, s2: true};
  var IDX = {edges: 0, s1: 1, s2: 2, inside: 3, outside: 4, center: 5,
             xy: 6, xyOut: 7, guides: 8};
  var bar = document.createElement('div');
  bar.style.cssText = 'font:14px sans-serif;padding:8px 12px;display:flex;' +
    'flex-wrap:wrap;gap:14px;align-items:center;';
  function el(tag, props) {
    var e = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) { e[k] = props[k]; });
    return e;
  }
  function field(text, child) {
    var l = el('label'); l.style.cssText = 'display:flex;gap:5px;align-items:center;';
    l.appendChild(document.createTextNode(text)); l.appendChild(child); return l;
  }
  function check(text, on, fn) {
    var c = el('input', {type: 'checkbox', checked: on});
    c.onchange = function () { fn(c.checked); update(); };
    var l = el('label'); l.style.cssText = 'display:flex;gap:5px;align-items:center;';
    l.appendChild(c); l.appendChild(document.createTextNode(text)); return l;
  }
  function select(options, value, fn) {
    var s = el('select');
    options.forEach(function (o, i) { s.appendChild(el('option', {value: i, textContent: o})); });
    s.value = value; s.onchange = function () { fn(+s.value); update(); }; return s;
  }
  var molSel = select(cfg.mols.map(function (m) { return m.title; }), 0,
    function (v) { st.mol = v; st.c = 0; fillCenters(); });
  var centerSel = el('select');
  centerSel.onchange = function () { st.c = +centerSel.value; update(); };
  function fillCenters() {
    centerSel.innerHTML = '';
    cfg.mols[st.mol].symbols.forEach(function (s, i) {
      centerSel.appendChild(el('option', {value: i, textContent: s + ' #' + i}));
    });
    centerSel.value = st.c;
  }
  var rSel = select(cfg.rmax.map(function (v) { return v + ' \\u00C5'; }), st.r, function (v) { st.r = v; });
  var lSel = select(cfg.layers.map(String), st.l, function (v) { st.l = v; });
  var prev = el('button', {textContent: '\u25c0'});
  var next = el('button', {textContent: '\u25b6'});
  function go(d) {
    var n = cfg.mols.length;
    st.mol = (st.mol + d + n) % n; st.c = 0; molSel.value = st.mol; fillCenters(); update();
  }
  prev.onclick = function () { go(-1); };
  next.onclick = function () { go(1); };
  if (cfg.mols.length > 1) {
    var molBox = el('span'); molBox.style.cssText = 'display:flex;gap:4px;align-items:center;';
    molBox.appendChild(prev); molBox.appendChild(molSel); molBox.appendChild(next);
    bar.appendChild(molBox);
  }
  bar.appendChild(field('centre atom', centerSel));
  bar.appendChild(field('r_max', rSel));
  bar.appendChild(field('layers', lSel));
  bar.appendChild(check('graph edges', true, function (v) { st.edges = v; }));
  bar.appendChild(check('r_max sphere', true, function (v) { st.s1 = v; }));
  bar.appendChild(check('layers \\u00D7 r_max sphere', true, function (v) { st.s2 = v; }));
  var info = el('span'); info.style.cssText = 'color:#555;';
  bar.appendChild(info);
  var key = el('span'); key.style.cssText = 'display:flex;gap:8px;align-items:center;';
  bar.appendChild(key);
  gd.parentNode.insertBefore(bar, gd);

  function env(d, r) {
    var u = d / r, p = cfg.p;
    if (u >= 1) return 0;
    return 1 - (p + 1) * (p + 2) / 2 * Math.pow(u, p) + p * (p + 2) * Math.pow(u, p + 1) -
      p * (p + 1) / 2 * Math.pow(u, p + 2);
  }
  function dist(P, a, b) {
    return Math.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1], P[a][2] - P[b][2]);
  }
  function hopColor(h) { return cfg.hopColors[Math.min(h, cfg.hopColors.length) - 1]; }

  function update() {
    var m = cfg.mols[st.mol], P = m.pos, N = P.length, c = st.c;
    var r = cfg.rmax[st.r], L = cfg.layers[st.l], hops = m.hops[st.r];
    var xi = [], yi = [], zi = [], ci = [], si = [], ti = [], di = [];
    var xo = [], yo = [], zo = [], so = [], to = [], dO = [];
    var px = [], py = [], pc = [], ox = [], oy = [];
    var nIn = 0, nBonded = 0, far = 0, j, h;
    for (j = 0; j < N; j++) {
      if (j === c) continue;
      h = hops[c * N + j];
      var d = dist(P, c, j);
      var text = m.symbols[j] + ' #' + j + '<br>distance ' + d.toFixed(2) + ' \\u00C5<br>' +
        (h > 0 ? h + (h === 1 ? ' hop' : ' hops') : 'not connected');
      if (h === 1) nBonded++;
      if (h >= 1 && h <= L) {
        nIn++; far = Math.max(far, d);
        xi.push(P[j][0]); yi.push(P[j][1]); zi.push(P[j][2]);
        ci.push(hopColor(h)); si.push(m.size[j]); ti.push(text); di.push(j);
        px.push(d); py.push(h); pc.push(hopColor(h));
      } else {
        xo.push(P[j][0]); yo.push(P[j][1]); zo.push(P[j][2]);
        so.push(m.size[j]); to.push(text); dO.push(j);
        if (h > L) { ox.push(d); oy.push(h); }
      }
    }
    // edges of the radius graph, shaded by the cutoff envelope
    var ex = [], ey = [], ez = [], ec = [];
    if (st.edges) {
      for (var a = 0; a < N; a++) for (var b = a + 1; b < N; b++) {
        var dd = dist(P, a, b);
        if (dd < r) {
          var w = env(dd, r);
          ex.push(P[a][0], P[b][0], null); ey.push(P[a][1], P[b][1], null);
          ez.push(P[a][2], P[b][2], null); ec.push(w, w, 0);
        }
      }
    }
    function sphere(R, on) {
      var S = cfg.sphere;
      return {
        x: on ? S.x.map(function (u) { return P[c][0] + R * u; }) : [],
        y: on ? S.y.map(function (u) { return P[c][1] + R * u; }) : [],
        z: on ? S.z.map(function (u) { return P[c][2] + R * u; }) : []
      };
    }
    var on1 = st.s1, on2 = st.s2 && L > 1;
    var s1 = sphere(r, on1), s2 = sphere(L * r, on2);
    var maxHop = 0, dmax = 0;
    for (j = 0; j < N; j++) { maxHop = Math.max(maxHop, hops[c * N + j]); dmax = Math.max(dmax, dist(P, c, j)); }
    var top = Math.max(maxHop, L) + 0.5, gx = [], gy = [];
    for (var k = 1; k <= Math.max(L, 1); k++) { gx.push(k * r, k * r, null); gy.push(0, top, null); }

    Plotly.restyle(gd, {x: [ex], y: [ey], z: [ez], 'line.color': [ec]}, [IDX.edges]);
    // Switched-off spheres are hidden with `visible` (a mesh with empty data is
    // not always cleared by the 3D renderer) and so also leave the legend.
    Plotly.restyle(gd, {x: [s1.x], y: [s1.y], z: [s1.z], visible: [on1]}, [IDX.s1]);
    Plotly.restyle(gd, {x: [s2.x], y: [s2.y], z: [s2.z], visible: [on2]}, [IDX.s2]);
    Plotly.restyle(gd, {x: [xi], y: [yi], z: [zi], 'marker.color': [ci], 'marker.size': [si],
      text: [ti], customdata: [di]}, [IDX.inside]);
    Plotly.restyle(gd, {x: [xo], y: [yo], z: [zo], 'marker.size': [so], text: [to],
      customdata: [dO]}, [IDX.outside]);
    Plotly.restyle(gd, {x: [[P[c][0]]], y: [[P[c][1]]], z: [[P[c][2]]],
      'marker.size': [[m.size[c] * 1.35]], customdata: [[c]],
      text: [[m.symbols[c] + ' #' + c + ' (centre)']]}, [IDX.center]);
    Plotly.restyle(gd, {x: [px], y: [py], 'marker.color': [pc]}, [IDX.xy]);
    Plotly.restyle(gd, {x: [ox], y: [oy]}, [IDX.xyOut]);
    Plotly.restyle(gd, {x: [gx], y: [gy]}, [IDX.guides]);
    Plotly.relayout(gd, {
      'title.text': m.title + ' \\u2013 centre ' + m.symbols[c] + ' #' + c +
        ', r_max ' + r + ' \\u00C5, ' + L + (L === 1 ? ' layer' : ' layers'),
      'xaxis.range': [0, Math.max(dmax, L * r) * 1.05], 'yaxis.range': [0.4, top],
      'yaxis.dtick': 1
    });
    info.textContent = nBonded + ' neighbours within r_max; ' + (nIn + 1) + ' of ' + N +
      ' atoms in the receptive field' + (nIn ? ' (farthest ' + far.toFixed(2) + ' \\u00C5, bound ' +
      (L * r).toFixed(2) + ' \\u00C5)' : '');
    key.innerHTML = '';
    function swatch(color, text) {
      var s = el('span'); s.style.cssText = 'display:flex;gap:4px;align-items:center;';
      var b = el('span'); b.style.cssText = 'width:11px;height:11px;border-radius:50%;background:' +
        color + ';display:inline-block;';
      s.appendChild(b); s.appendChild(document.createTextNode(text)); key.appendChild(s);
    }
    swatch(cfg.centerColor, 'centre');
    for (var q = 1; q <= L; q++) swatch(hopColor(q), q + (q === 1 ? ' hop' : ' hops'));
    swatch(cfg.outsideColor, 'outside the receptive field');
  }
  document.addEventListener('keydown', function (e) {
    if (cfg.mols.length < 2 || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') go(-1);
    if (e.key === 'ArrowRight') go(1);
  });
  gd.on('plotly_click', function (ev) {
    var p = ev.points && ev.points[0];
    if (p && [IDX.inside, IDX.outside, IDX.center].indexOf(p.curveNumber) >= 0 &&
        p.customdata !== undefined) {
      st.c = +p.customdata; centerSel.value = st.c; update();
    }
  });
  fillCenters();
  update();
})();
"""


def polynomial_cutoff(distance: Any, r_max: float, p: int = POLYNOMIAL_CUTOFF):
    """Evaluate MACE's smooth cutoff envelope (1 at 0, 0 at ``r_max``).

    Parameters
    ----------
    distance : array-like
        Distances in Angstrom.
    r_max : float
        Cutoff radius.
    p : int
        Exponent of the polynomial (MACE's ``num_polynomial_cutoff``).

    Returns
    -------
    numpy.ndarray
        The envelope at each distance; zero beyond ``r_max``.
    """
    u = np.asarray(distance, dtype=float) / r_max
    value = (
        1.0
        - (p + 1.0) * (p + 2.0) / 2.0 * u**p
        + p * (p + 2.0) * u ** (p + 1.0)
        - p * (p + 1.0) / 2.0 * u ** (p + 2.0)
    )
    return np.where(u < 1.0, value, 0.0)


def hop_matrix(positions: Any, r_max: float) -> np.ndarray:
    """Return the number of hops between all atom pairs of the radius graph.

    Parameters
    ----------
    positions : array-like
        Atom positions in Angstrom, shape ``[N, 3]``.
    r_max : float
        Cutoff radius: atoms closer than this are connected.

    Returns
    -------
    numpy.ndarray
        Integer matrix ``[N, N]``: 0 on the diagonal, the shortest number of
        edges between two atoms, or -1 when they are not connected.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    pos = np.asarray(positions, dtype=float)
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    adjacency = (dist < r_max) & ~np.eye(len(pos), dtype=bool)
    hops = shortest_path(csr_matrix(adjacency), unweighted=True)
    return np.where(np.isfinite(hops), hops, -1).astype(int)


def unit_sphere(n_lat: int = 14, n_lon: int = 24) -> dict[str, list]:
    """Return the vertices and triangles of a unit sphere for a ``Mesh3d``.

    Parameters
    ----------
    n_lat : int
        Number of latitude rings.
    n_lon : int
        Number of longitude steps.

    Returns
    -------
    dict[str, list]
        ``x``, ``y``, ``z`` (vertices) and ``i``, ``j``, ``k`` (triangles).
    """
    theta = np.linspace(0, np.pi, n_lat + 1)
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    x, y, z = (
        np.sin(tt) * np.cos(pp),
        np.sin(tt) * np.sin(pp),
        np.cos(tt),
    )
    tri_i, tri_j, tri_k = [], [], []
    for a in range(n_lat):
        for b in range(n_lon):
            p00 = a * n_lon + b
            p01 = a * n_lon + (b + 1) % n_lon
            p10 = (a + 1) * n_lon + b
            p11 = (a + 1) * n_lon + (b + 1) % n_lon
            tri_i += [p00, p01]
            tri_j += [p10, p11]
            tri_k += [p01, p10]
    return {
        "x": x.ravel().tolist(),
        "y": y.ravel().tolist(),
        "z": z.ravel().tolist(),
        "i": tri_i,
        "j": tri_j,
        "k": tri_k,
    }


def mace_receptive_field_page(
    molecules: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    r_max: Sequence[float] = (2.0, 3.0, 4.0, 5.0, 6.0),
    layers: Sequence[int] = (1, 2, 3, 4),
    size_scale: float = 1.0,
    offline: bool = False,
) -> Path:
    """Write the interactive receptive-field page for a list of molecules.

    Parameters
    ----------
    molecules : sequence of dict
        Each with ``numbers``, ``positions`` and ``title``.
    path : str or Path
        Output HTML file.
    r_max : sequence of float
        Cutoff radii (Angstrom) offered in the page; the first is shown first.
    layers : sequence of int
        Numbers of interaction layers offered in the page; the first is shown
        first.
    size_scale : float
        Factor on the atom marker sizes.
    offline : bool
        Embed the Plotly library so the page opens without internet access.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    r_max, layers = [float(r) for r in r_max], [int(n) for n in layers]
    mols = []
    for molecule in molecules:
        numbers = np.asarray(molecule["numbers"], dtype=int)
        pos = np.asarray(molecule["positions"], dtype=float)
        mols.append(
            {
                "title": molecule["title"],
                "symbols": [chemical_symbols[z] for z in numbers],
                "pos": pos.tolist(),
                "size": atom_size(numbers, size_scale).tolist(),
                "hops": {
                    str(i): hop_matrix(pos, radius).ravel().tolist()
                    for i, radius in enumerate(r_max)
                },
            }
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.64, 0.36],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        horizontal_spacing=0.06,
    )
    empty3 = {"x": [], "y": [], "z": []}
    sphere = unit_sphere()
    traces = [
        go.Scatter3d(
            **empty3,
            mode="lines",
            line={
                "color": [],
                "colorscale": [[0, "#e3e6ec"], [1, "#2b2b33"]],
                "cmin": 0,
                "cmax": 1,
                "width": 5,
            },
            name="graph edges (shade = cutoff envelope)",
            hoverinfo="skip",
        ),
        go.Mesh3d(
            **empty3,
            i=sphere["i"],
            j=sphere["j"],
            k=sphere["k"],
            color="#4c78a8",
            opacity=0.10,
            name="sphere of radius r_max",
            showlegend=True,
            hoverinfo="skip",
        ),
        go.Mesh3d(
            **empty3,
            i=sphere["i"],
            j=sphere["j"],
            k=sphere["k"],
            color="#f58518",
            opacity=0.06,
            name="sphere of radius layers × r_max",
            showlegend=True,
            hoverinfo="skip",
        ),
        go.Scatter3d(
            **empty3,
            mode="markers",
            marker={
                "size": [],
                "color": [],
                "line": {"color": "#333333", "width": 1.5},
            },
            name="in the receptive field",
            hoverinfo="text",
        ),
        go.Scatter3d(
            **empty3,
            mode="markers",
            marker={
                "size": [],
                "color": OUTSIDE_COLOR,
                "line": {"color": "#333333", "width": 1.0},
            },
            name="outside the receptive field",
            hoverinfo="text",
        ),
        go.Scatter3d(
            **empty3,
            mode="markers",
            marker={
                "size": [],
                "color": CENTER_COLOR,
                "symbol": "circle",
                "line": {"color": "#e63946", "width": 4},
            },
            name="centre atom",
            hoverinfo="text",
        ),
    ]
    for trace in traces:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 9, "line": {"color": "#333333", "width": 1}},
            name="atoms in the field",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={
                "size": 9,
                "color": OUTSIDE_COLOR,
                "line": {"color": "#333333", "width": 1},
            },
            name="atoms beyond the layers",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line={"color": "#f58518", "dash": "dot", "width": 1.5},
            name="k × r_max",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title={"text": "", "x": 0.02},
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x (Å)"},
            "yaxis": {"title": "y (Å)"},
            "zaxis": {"title": "z (Å)"},
        },
        xaxis={"title": "distance from the centre atom (Å)"},
        yaxis={"title": "hops from the centre atom", "type": "linear"},
        legend={"itemsizing": "constant", "y": 0.98},
        margin={"l": 0, "r": 10, "t": 60, "b": 40},
        height=720,
    )
    config = {
        "mols": mols,
        "rmax": r_max,
        "layers": layers,
        "start": {"r": 0, "l": 0},  # the first listed values are shown first
        "sphere": {k: sphere[k] for k in ("x", "y", "z")},
        "p": POLYNOMIAL_CUTOFF,
        "hopColors": HOP_COLORS,
        "centerColor": CENTER_COLOR,
        "outsideColor": OUTSIDE_COLOR,
    }
    script = PAGE_JS.replace("__CONFIG__", json.dumps(config))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        fig.to_html(
            include_plotlyjs=True if offline else "cdn", post_script=script
        ),
        encoding="utf-8",
    )
    return path


def view_mace_receptive_field(
    dataset: str,
    indices: Sequence[int],
    save_dir: str | Path,
    base_name: str,
    *,
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Path:
    """Save one page showing the receptive field of MACE for OpenQDC molecules.

    The page ``<save_dir>/<base_name>.html`` lets you choose the molecule, the
    centre atom (menu, or click an atom), the cutoff radius and the number of
    layers, and shows which atoms the centre's energy can depend on. It is
    computed from the geometry alone; no model is used.

    Parameters
    ----------
    dataset : str
        OpenQDC dataset name, e.g. ``"QM9"``.
    indices : sequence of int
        Indices of the molecules in the OpenQDC dataset.
    save_dir : str or Path
        Folder for the HTML file (created if missing).
    base_name : str
        File name without the ``.html`` suffix.
    cache_dir : str, optional
        OpenQDC cache directory; default as in the training scripts.
    **kwargs : Any
        Passed to :func:`mace_receptive_field_page` (``r_max``, ``layers``,
        ``size_scale``, ``offline``).

    Returns
    -------
    pathlib.Path
        The written file.
    """
    source, indices = resolve_molecules(dataset, indices, base_name, cache_dir)
    molecules = []
    for index in indices:
        numbers, _, positions, _ = source.get(index, "energies", 0)
        molecules.append(
            {
                "numbers": numbers,
                "positions": positions,
                "title": f"{source.name} #{index}: {len(numbers)} atoms",
            }
        )
    return mace_receptive_field_page(
        molecules, Path(save_dir) / f"{base_name}.html", **kwargs
    )


def main() -> None:
    """Command line entry: write the receptive-field page of OpenQDC molecules."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="e.g. QM9")
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--base-name", default="mace_receptive_field")
    parser.add_argument(
        "--r-max",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0, 5.0, 6.0],
        help="cutoff radii offered in the page (Angstrom)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="numbers of interaction layers offered in the page",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    path = view_mace_receptive_field(
        args.dataset,
        args.indices,
        args.save_dir,
        args.base_name,
        cache_dir=args.cache_dir,
        r_max=args.r_max,
        layers=args.layers,
        offline=args.offline,
    )
    print(f"Wrote {path}; open it in a browser.")


if __name__ == "__main__":
    main()


__all__ = [
    "hop_matrix",
    "mace_receptive_field_page",
    "polynomial_cutoff",
    "view_mace_receptive_field",
]
