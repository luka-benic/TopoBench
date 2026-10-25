"""Source-unit regression checks through the actual OpenQDC readers."""

import hashlib
import shutil
from importlib import import_module
from pathlib import Path

import numpy as np
import openqdc
import pytest
from ase.io import read
from omegaconf import OmegaConf
from openqdc.datasets import structure

from test.utils.test_openqdc_export import export_xyz as export_xyz
from test.utils.test_openqdc_export import make_args
from topobench.data.datasets.openqdc_dataset import OpenQDCDataset
from topobench.data.utils import openqdc_utils as utils

base = import_module("openqdc.datasets.base")
pytestmark = pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:"
    "DeprecationWarning:openqdc.methods.atom_energies"
)

# QM9 methane (row 4355) and QM7b row 0, checked against the source archives.
REFERENCE_ENERGIES = {
    "QM9": (-25400.912063377767, -1101.487799),
    "QM7b": (-433379.127214, -18793.097649),
}


@pytest.fixture
def qmx_cache(tmp_path):
    """Write a small upstream-format cache with synthetic methane geometries."""

    def build(name):
        cache = tmp_path / "raw" / name
        cls = utils._prepare_class(name, str(cache))
        ds = cls.no_init()
        molecule = np.array(
            [
                [6, 0, 0, 0, 0],
                [1, 0, 1.2, 1.2, 1.2],
                [1, 0, 1.2, -1.2, -1.2],
                [1, 0, -1.2, 1.2, -1.2],
                [1, 0, -1.2, -1.2, 1.2],
            ],
            dtype=np.float32,
        )
        energies = REFERENCE_ENERGIES[name][0] + np.arange(8) * 0.1
        ds.save_preprocess(
            {
                "atomic_inputs": np.tile(molecule, (8, 1)),
                "position_idx_range": np.stack(
                    [np.arange(8), np.arange(1, 9)], axis=1
                ).astype(np.int32)
                * 5,
                "energies": np.repeat(
                    energies[:, None], len(ds.energy_methods), axis=1
                ),
                "name": np.array(["C"] * 8),
                "subset": np.array([name.lower()] * 8),
                "n_atoms": np.full(8, 5, dtype=np.int32),
            }
        )
        return cache

    return build


@pytest.mark.parametrize("name", REFERENCE_ENERGIES)
@pytest.mark.parametrize("first_download", [False, True])
def test_first_load_and_cache_reuse(
    name, first_download, qmx_cache, tmp_path, monkeypatch
):
    cache = qmx_cache(name)
    upstream = cache / "qmx" / "preprocessed" / name.lower()
    before = {
        p.name: hashlib.sha256(p.read_bytes()).digest()
        for p in upstream.iterdir()
    }
    downloads = []
    if first_download:
        cache = tmp_path / "fresh"

        def copy_exists(path):
            return (upstream / Path(path).name).is_file()

        def pull_locally(path, overwrite=False):
            if not Path(path).exists() or overwrite:
                shutil.copyfile(upstream / Path(path).name, path)
                downloads.append(Path(path).name)
            return path

        monkeypatch.setattr(base, "copy_exists", copy_exists)
        monkeypatch.setattr(structure, "pull_locally", pull_locally)

    for _ in range(2):
        source = utils.OpenQDCSource(name, str(cache))
        assert source.get(0)[3] == pytest.approx(
            REFERENCE_ENERGIES[name][1], abs=0.003, rel=0
        )
        # Position conversion is still bohr to Angstrom; no second energy pass.
        np.testing.assert_allclose(
            source.get(0)[2][1], np.full(3, 1.2 * 0.52917721092), rtol=1e-6
        )
        assert source.get(0, "per_atom_formation_energies")[
            3
        ] == pytest.approx(source.get(0, "formation_energies")[3] / 5)
    after = cache / "qmx" / "preprocessed" / name.lower()
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).digest()
        for p in after.iterdir()
    }
    assert len(downloads) == (4 if first_download else 0)
    assert getattr(openqdc, name).__energy_unit__ == "hartree"


def test_other_readers_are_unchanged(tmp_path):
    for name in ["Alchemy", "QM8", "MultixcQM9"]:
        assert utils._prepare_class(name, str(tmp_path)) is getattr(
            openqdc, name
        )
        assert utils.energy_cache_suffix(name) == ""


def test_upstream_fix_is_not_applied_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(openqdc.QM9, "__energy_unit__", "kcal/mol")
    monkeypatch.setattr(utils, "version", lambda name: "0.1.3")
    assert utils._prepare_class("QM9", str(tmp_path)) is openqdc.QM9


def test_unverified_upstream_version_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "version", lambda name: "0.1.3")
    with pytest.raises(RuntimeError, match="Verify QM9 source energy units"):
        utils._prepare_class("QM9", str(tmp_path))


@pytest.mark.parametrize("name", REFERENCE_ENERGIES)
@pytest.mark.parametrize("baseline", ["none", "per_element"])
def test_graphs_and_xyz_use_the_same_corrected_targets(
    name, baseline, qmx_cache, export_xyz, tmp_path
):
    cache = qmx_cache(name)
    parameters = OmegaConf.create(
        {
            "openqdc_cache_dir": str(cache),
            "atom_types": [1, 6],
            "baseline": baseline,
            "valid_fraction": 0.25,
            "test_fraction": 0.25,
            "split_seed": 0,
        }
    )
    root = tmp_path / "graphs"
    dataset = OpenQDCDataset(str(root), name, parameters)
    legacy = Path(
        dataset.processed_root.replace(utils.energy_cache_suffix(name), "")
    )
    (legacy / "processed").mkdir(parents=True)
    (legacy / "processed/data.pt").write_bytes(
        b"old incorrectly scaled graphs"
    )
    args = make_args(
        tmp_path / "xyz",
        dataset=name,
        cache_dir=str(cache),
        max_samples=None,
        valid_fraction=0.25,
        test_fraction=0.25,
    )
    paths = export_xyz.ensure_export(args)
    for split, path in paths.items():
        frames = read(path, index=":")
        ids = dataset.split_idx[split].tolist()
        assert [frame.info["openqdc_index"] for frame in frames] == ids
        for frame in frames:
            graph = dataset[frame.info["openqdc_index"]]
            assert frame.info["REF_energy"] == graph.energy.item()
    assert dataset[0].energy.item() == pytest.approx(
        REFERENCE_ENERGIES[name][1], abs=0.003, rel=0
    )
    train = dataset.split_idx["train"].tolist()
    offset = (
        np.mean([dataset[i].energy.item() for i in train])
        if baseline == "per_element"
        else 0
    )
    for graph in dataset:
        assert graph.y.item() == pytest.approx(
            graph.energy.item() - offset, abs=0.003, rel=0
        )
    # A subsequent load must select the corrected cache, not the old one.
    reloaded = OpenQDCDataset(str(root), name, parameters)
    assert not reloaded.was_processed
    assert reloaded[0].energy.item() == dataset[0].energy.item()
