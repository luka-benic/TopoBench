"""Tests of the MACE receptive-field viewer (geometry only, no model)."""

import numpy as np
import pytest

from topobench.utils.mace_viewer import (
    hop_matrix,
    mace_receptive_field_page,
    polynomial_cutoff,
)

# A chain of five carbon atoms 1.5 A apart.
CHAIN_Z = [6, 6, 6, 6, 6]
CHAIN_POS = [[1.5 * i, 0.0, 0.0] for i in range(5)]


def test_hop_matrix_counts_edges_of_the_radius_graph():
    hops = hop_matrix(CHAIN_POS, 1.6)  # only neighbours are connected
    assert hops[0].tolist() == [0, 1, 2, 3, 4]
    assert hops[2].tolist() == [2, 1, 0, 1, 2]
    wide = hop_matrix(CHAIN_POS, 3.2)  # second neighbours too
    assert wide[0].tolist() == [0, 1, 1, 2, 2]
    far = hop_matrix(CHAIN_POS + [[40.0, 0.0, 0.0]], 1.6)
    assert far[0, 5] == -1  # not connected


def test_atoms_k_hops_away_are_within_k_times_r_max():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 8, size=(30, 3))
    for r_max in (1.5, 2.5, 4.0):
        hops = hop_matrix(pos, r_max)
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        reached = hops >= 0
        assert (dist[reached] <= hops[reached] * r_max + 1e-9).all()


def test_cutoff_envelope_matches_the_one_inside_mace():
    pytest.importorskip("mace")
    torch = pytest.importorskip("torch")
    from mace.modules.radial import PolynomialCutoff

    d = np.linspace(0.0, 6.0, 25)
    reference = PolynomialCutoff(r_max=5.0, p=5)(torch.tensor(d)).numpy()
    assert np.allclose(polynomial_cutoff(d, 5.0, 5), reference)
    assert polynomial_cutoff(5.0, 5.0) == 0.0
    assert polynomial_cutoff(0.0, 5.0) == 1.0


def test_page_holds_the_hops_for_every_cutoff(tmp_path):
    path = mace_receptive_field_page(
        [{"numbers": CHAIN_Z, "positions": CHAIN_POS, "title": "chain"}],
        tmp_path / "out" / "page.html",
        r_max=[1.6, 3.2],
        layers=[1, 2],
    )
    flat = path.read_text().replace(" ", "")
    assert '"hops":{"0":[' in flat and '"1":[' in flat
    assert '"rmax":[1.6,3.2]' in flat and '"layers":[1,2]' in flat
    assert "model" not in flat.lower().replace("molecule", "")  # no model data


def test_page_does_not_import_mace_or_torch_models():
    from pathlib import Path

    import topobench.utils.mace_viewer as module

    source = Path(module.__file__).read_text()
    assert "import mace" not in source and "from mace" not in source
    assert "torch" not in source and "e3nn" not in source


def test_page_has_arrow_buttons_and_keys_for_the_molecules(tmp_path):
    molecules = [
        {"numbers": CHAIN_Z, "positions": CHAIN_POS, "title": f"chain {i}"}
        for i in range(3)
    ]
    text = mace_receptive_field_page(
        molecules, tmp_path / "p.html"
    ).read_text()
    assert "\u25c0" in text and "\u25b6" in text  # the two arrow buttons
    assert "ArrowLeft" in text and "ArrowRight" in text
    assert '"title": "chain 2"' in text or '"title":"chain2"' in text.replace(
        " ", ""
    )


def test_switched_off_spheres_are_hidden_not_just_emptied(tmp_path):
    text = mace_receptive_field_page(
        [{"numbers": CHAIN_Z, "positions": CHAIN_POS, "title": "c"}],
        tmp_path / "p.html",
    ).read_text()
    assert "visible: [on1]" in text and "visible: [on2]" in text


def test_any_lists_work_and_the_first_values_are_shown_first(tmp_path):
    """Nothing is required of the lists (no 5 A, no 2 layers): they are shown as given."""
    molecules = [{"numbers": CHAIN_Z, "positions": CHAIN_POS, "title": "c"}]
    path = mace_receptive_field_page(
        molecules, tmp_path / "p.html", r_max=[3, 2, 4], layers=[3, 1]
    )
    flat = path.read_text().replace(" ", "")
    assert '"rmax":[3.0,2.0,4.0]' in flat and '"layers":[3,1]' in flat
    assert '"start":{"r":0,"l":0}' in flat


def test_atoms_are_coloured_by_element_with_the_jmol_colours(tmp_path):
    text = mace_receptive_field_page(
        [
            {
                "numbers": [6, 8, 1],
                "positions": [[0, 0, 0], [1.4, 0, 0], [0, 1.0, 0]],
                "title": "co",
            }
        ],
        tmp_path / "p.html",
    ).read_text()
    flat = text.replace(" ", "")
    # carbon grey, oxygen red, hydrogen white (Jmol), in the order of the atoms
    assert (
        '"colors":["rgb(144,144,144)","rgb(255,13,13)","rgb(255,255,255)"]'
        in flat
    )
    assert '"z":[6,8,1]' in flat
    assert "colourby" in flat.replace("colour by", "colourby")  # the mode menu
    assert "element (ring = hops)" in text
