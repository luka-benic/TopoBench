"""Tests of the OpenQDC molecule viewer and the shared edge construction."""

import numpy as np
import pytest

from topobench.data.utils.openqdc_utils import molecular_edges
from topobench.utils.openqdc_viewer import (
    atom_size,
    view_atoms,
    view_molecules,
)

# Water: O at the origin, two H at 0.96 Angstrom; the H-H distance is 1.51.
WATER_Z = [8, 1, 1]
WATER_POS = [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]


def undirected(edge_index):
    return {tuple(sorted(e)) for e in edge_index.t().tolist()}


def test_bond_rule_connects_oh_not_hh():
    edges = molecular_edges(WATER_POS, WATER_Z, "bonds")
    assert undirected(edges) == {(0, 1), (0, 2)}
    assert edges.shape[1] == 4  # both directions, no self-loops


def test_radius_rule_depends_only_on_the_cutoff():
    assert undirected(
        molecular_edges(WATER_POS, WATER_Z, "radius", cutoff=1.2)
    ) == {
        (0, 1),
        (0, 2),
    }
    assert (
        len(
            undirected(
                molecular_edges(WATER_POS, WATER_Z, "radius", cutoff=2.0)
            )
        )
        == 3
    )
    with pytest.raises(ValueError):
        molecular_edges(WATER_POS, WATER_Z, "nope")


def test_atom_size_grows_with_atomic_number():
    sizes = atom_size([1, 6, 8, 79])
    assert list(sizes) == sorted(sizes)
    assert sizes[0] == pytest.approx(7.0)


def test_figure_has_one_trace_per_element_and_edge_set():
    fig = view_atoms(WATER_Z, WATER_POS, edges="bonds")
    names = [t.name for t in fig.data]
    assert len(fig.data) == 2 + 2  # two edge sets, O and H
    assert names[0].startswith("cutoff edges") and names[1].startswith(
        "bond edges"
    )
    assert any(n.startswith("O (Z=8") for n in names)
    assert any(n.startswith("H (Z=1, n=2") for n in names)
    assert fig.data[0].visible == "legendonly"  # cutoff edges off at first
    assert fig.data[1].visible is True


def test_buttons_switch_the_edge_sets():
    fig = view_atoms(WATER_Z, WATER_POS)
    buttons = {
        b.label: b.args[0]["visible"]
        for b in fig.layout.updatemenus[0].buttons
    }
    assert buttons["No edges"][:2] == [False, False]
    assert buttons["Bonds only"][:2] == [False, True]
    assert buttons["Cutoff only"][:2] == [True, False]
    assert all(v[2:] == [True, True] for v in buttons.values())


def test_edge_segments_match_the_edge_count():
    fig = view_atoms(WATER_Z, WATER_POS)
    bond_trace = fig.data[1]
    assert len(bond_trace.x) == 3 * 2  # two bonds, each (start, end, None)
    assert bond_trace.x[2] is None
    assert np.isclose(bond_trace.x[1], 0.96)


def test_wrong_shapes_and_modes_are_rejected():
    with pytest.raises(ValueError):
        view_atoms(WATER_Z, WATER_POS[:2])
    with pytest.raises(ValueError):
        view_atoms(WATER_Z, WATER_POS, edges="all")


def test_cutoff_and_bond_lines_are_solid():
    fig = view_atoms(WATER_Z, WATER_POS)
    assert all(t.line.dash in (None, "solid") for t in fig.data[:2])


def test_bond_scale_and_cutoff_are_chosen_by_the_caller():
    fig = view_atoms(WATER_Z, WATER_POS, bond_scale=2.5, cutoff=1.2)
    assert (
        fig.data[1].name == "bond edges (2.5 × covalent radii) (3)"
    )  # H-H now bonded too
    assert fig.data[0].name.startswith("cutoff edges (< 1.2 Å) (2)")


class FakeSource:
    name = "Fake"
    method_names = ["energy"]

    def __len__(self):
        return 10

    def get(self, i, target, method):
        pos = np.array(WATER_POS) + 0.01 * i
        return np.array(WATER_Z), np.zeros(3), pos, -float(i)


def test_view_molecules_writes_one_page_with_all_molecules(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "topobench.utils.openqdc_viewer.method_names_for",
        lambda name: ["energy"],
    )
    path = view_molecules(FakeSource(), [7, 2, 4], tmp_path / "out", "water")
    assert path == tmp_path / "out" / "water.html"
    text = path.read_text()
    for index in (7, 2, 4):
        assert f"Fake #{index}: 3 atoms" in text
    assert list((tmp_path / "out").iterdir()) == [path]  # a single file


def test_view_molecules_checks_everything_before_writing(tmp_path):
    for bad in ([], [1, 1], [3, 99]):
        with pytest.raises((ValueError, IndexError)):
            view_molecules(FakeSource(), bad, tmp_path / "out", "water")
    with pytest.raises(ValueError):
        view_molecules(FakeSource(), [1], tmp_path, "sub/water")
    assert not (tmp_path / "out").exists()


# Four hydrogens that are all bonded to each other form a tetrahedron.
TETRA_Z = [1, 1, 1, 1]
TETRA_POS = [[0, 0, 0], [0.7, 0, 0], [0.35, 0.6, 0], [0.35, 0.2, 0.57]]


def test_models_sharing_a_lifting_share_it_and_graph_models_have_none():
    from topobench.utils.openqdc_liftings import lifting_key, model_liftings

    assert model_liftings("graph/gcn") == []
    assert lifting_key(model_liftings("simplicial/scn")) == lifting_key(
        model_liftings("simplicial/san")
    )
    assert lifting_key(model_liftings("simplicial/scn")) != lifting_key(
        model_liftings("cell/cwn")
    )


def test_clique_lifting_finds_triangles_and_the_tetrahedron():
    from topobench.utils.openqdc_liftings import lift_to_cells, model_liftings

    cells = lift_to_cells(TETRA_Z, TETRA_POS, model_liftings("simplicial/scn"))
    assert len(cells["rank 2"]) == 4
    assert cells["rank 3"] == [(0, 1, 2, 3)]


def test_cycle_lifting_finds_the_ring_and_hypergraph_gives_hyperedges():
    from topobench.utils.openqdc_liftings import lift_to_cells, model_liftings

    angles = np.arange(3) * 2 * np.pi / 3
    z = [6, 6, 6]
    pos = np.c_[0.87 * np.cos(angles), 0.87 * np.sin(angles), np.zeros(3)]
    ring = lift_to_cells(z, pos, model_liftings("cell/cwn"))
    assert ring["rank 2"] == [(0, 1, 2)]
    hyper = lift_to_cells(z, pos, model_liftings("hypergraph/unignn2"))
    assert len(hyper["hyperedges"]) == 3  # one k-hop neighbourhood per atom


def test_lifted_cells_are_drawn_in_light_pink():
    from topobench.utils.openqdc_viewer import LIFT_COLOR, _lifting_traces

    cells = {"rank 2": [(0, 1, 2)], "rank 3": [(0, 1, 2, 3)]}
    traces = _lifting_traces(
        np.array(TETRA_POS), cells, "demo", {"kind": "lifting"}, True
    )
    assert LIFT_COLOR == "lightpink"
    assert len(traces) == 2 and all(t.color == "lightpink" for t in traces)
    assert len(traces[1].i) == 4  # tetrahedron: four faces


def test_page_lists_liftings_and_switches_between_them(tmp_path, monkeypatch):
    class TetraSource(FakeSource):
        def get(self, i, target, method):
            return np.array(TETRA_Z), np.zeros(4), np.array(TETRA_POS), -1.0

    monkeypatch.setattr(
        "topobench.utils.openqdc_viewer.method_names_for",
        lambda name: ["energy"],
    )
    path = view_molecules(
        TetraSource(),
        [0, 1],
        tmp_path,
        "tetra",
        models=["simplicial/scn", "simplicial/san", "cell/cwn", "graph/gcn"],
    )
    text = path.read_text()
    assert "simplicial/scn, simplicial/san: SimplicialCliqueLifting" in text
    assert "cell/cwn: CellCycleLifting" in text
    assert "lightpink" in text and "graph/gcn:" not in text


def benzene():
    angles = np.arange(6) * np.pi / 3
    carbon = np.c_[1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)]
    hydrogen = np.c_[2.48 * np.cos(angles), 2.48 * np.sin(angles), np.zeros(6)]
    return np.array([6] * 6 + [1] * 6), np.vstack([carbon, hydrogen])


def test_liftings_follow_the_graph_that_is_lifted():
    from topobench.utils.openqdc_liftings import lift_to_cells, model_liftings

    z, pos = benzene()
    clique = model_liftings("simplicial/scn")
    on_bonds = lift_to_cells(z, pos, clique, graph_edges="bonds")
    on_cutoff = lift_to_cells(z, pos, clique, graph_edges="radius", cutoff=2.6)
    assert (
        on_bonds["rank 2"] == []
    )  # the bond graph of benzene has no triangle
    assert len(on_cutoff["rank 2"]) > 0  # meta C-C pairs are within 2.6 A


def test_cutoff_and_bond_scale_can_be_lists(tmp_path, monkeypatch):
    class BenzeneSource(FakeSource):
        def get(self, i, target, method):
            z, pos = benzene()
            return z, np.zeros(len(z)), pos, -1.0

    monkeypatch.setattr(
        "topobench.utils.openqdc_viewer.method_names_for",
        lambda name: ["energy"],
    )
    path = view_molecules(
        BenzeneSource(),
        [0],
        tmp_path,
        "benzene",
        models=["simplicial/scn"],
        bond_scale=[1.1, 1.3],
        cutoff=[1.6, 2.6],
    )
    text = path.read_text()
    for name in (
        "bond edges (1.1 × covalent radii)",
        "bond edges (1.3 × covalent radii)",
        "cutoff edges (&lt; 1.6 Å)",
        "cutoff edges (&lt; 2.6 Å)",
    ):
        assert name.replace("&lt;", "\\u003c") in text or name in text
    flat = text.replace(" ", "")
    assert '"graph":"bonds"' in flat and '"graph":"radius"' in flat
    assert (
        "no cells of rank 2 or higher" in text
    )  # clique complex of the bonds
    assert '"bondScales":["1.1","1.3"]' in text.replace(" ", "")


def test_value_lists_must_be_distinct_and_positive():
    from topobench.utils.openqdc_viewer import _as_values

    assert _as_values(1.3, "x") == [1.3]
    for bad in ([], [1.0, 1.0], [0.0], -1):
        with pytest.raises(ValueError):
            _as_values(bad, "x")
