"""Apply the liftings of TopoBench models to a molecule and read their cells.

A TopoBench model for higher-order domains (simplicial, cell, hypergraph,
combinatorial) does not run on the molecular graph itself: the dataset graph is
first *lifted* to that domain. Which lifting a model uses is decided by the
Hydra configs (``configs/transforms``), exactly as in a training run. This
module resolves those liftings for a model name such as ``"simplicial/scn"``,
applies them to one molecule, and returns the resulting cells as tuples of atom
indices, ready to be drawn.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from topobench.data.utils.openqdc_utils import PROJECT_ROOT, molecular_edges

log = logging.getLogger(__name__)

# The dataset config only fixes the data domain (graph) of the composed run
# config; the lifting of a model does not depend on which OpenQDC dataset it is.
DATASET_STAND_IN = "graph/openqdc_qm9"


def model_liftings(
    model: str, config_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Resolve the lifting transforms a TopoBench model uses on OpenQDC graphs.

    The run config is composed like a training run would (``dataset`` and
    ``model`` chosen), so the model's own and default liftings are honoured.
    Transforms that are not liftings (feature encodings, positional
    encodings) are ignored: they do not change the structure.

    Parameters
    ----------
    model : str
        Model config name, e.g. ``"simplicial/scn"`` or ``"hypergraph/unignn2"``.
    config_dir : str or Path, optional
        The TopoBench ``configs`` folder; default: the one of this repository.

    Returns
    -------
    list[dict[str, Any]]
        The lifting transform configs in the order they are applied; empty for
        models that run on the plain graph.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    import topobench  # noqa: F401  (registers the config resolvers)

    config_dir = Path(config_dir or PROJECT_ROOT / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(str(config_dir.resolve()), version_base="1.3"):
        cfg = compose(
            config_name="run",
            overrides=[f"dataset={DATASET_STAND_IN}", f"model={model}"],
        )
    transforms = cfg.get("transforms")
    if not transforms:
        return []
    return [
        {k: v for k, v in item.items() if k != "preprocessor_device"}
        for item in OmegaConf.to_container(transforms, resolve=True).values()
        if isinstance(item, dict) and item.get("transform_type") == "lifting"
    ]


def lifting_key(liftings: list[dict[str, Any]]) -> str:
    """Return a stable identifier of a lifting pipeline.

    Parameters
    ----------
    liftings : list[dict[str, Any]]
        Lifting configs as returned by :func:`model_liftings`.

    Returns
    -------
    str
        A JSON string; equal pipelines give equal keys.
    """
    return json.dumps(liftings, sort_keys=True, default=str)


def lifting_label(liftings: list[dict[str, Any]]) -> str:
    """Name a lifting pipeline for a legend or menu.

    Parameters
    ----------
    liftings : list[dict[str, Any]]
        Lifting configs as returned by :func:`model_liftings`.

    Returns
    -------
    str
        The transform names joined by ``+``.
    """
    return " + ".join(str(t.get("transform_name")) for t in liftings)


def lift_molecule(
    numbers: Any,
    positions: Any,
    liftings: list[dict[str, Any]],
    graph_edges: str = "bonds",
    bond_scale: float = 1.3,
    cutoff: float = 5.0,
) -> Data:
    """Apply lifting transforms to the graph of one molecule.

    Parameters
    ----------
    numbers : array-like
        Atomic numbers.
    positions : array-like
        Positions in Angstrom.
    liftings : list[dict[str, Any]]
        Lifting configs as returned by :func:`model_liftings`.
    graph_edges : str
        Rule that builds the input graph, ``"bonds"`` or ``"radius"``, as the
        ``edge_method`` of the dataset.
    bond_scale : float
        Factor on the covalent radii for ``graph_edges="bonds"``.
    cutoff : float
        Distance cutoff in Angstrom for ``graph_edges="radius"``.

    Returns
    -------
    torch_geometric.data.Data
        The lifted data (incidence matrices of the target domain).
    """
    from topobench.transforms.data_transform import DataTransform

    numbers = np.asarray(numbers, dtype=int)
    species = sorted(set(numbers.tolist()))
    x = torch.zeros(len(numbers), len(species))
    x[
        torch.arange(len(numbers)),
        torch.tensor([species.index(z) for z in numbers]),
    ] = 1
    data = Data(
        x=x,
        pos=torch.as_tensor(np.asarray(positions), dtype=torch.float32),
        edge_index=molecular_edges(
            positions, numbers, graph_edges, bond_scale, cutoff
        ),
        num_nodes=len(numbers),
    )
    for config in liftings:
        data = DataTransform(**config)(data)
    return data


def _dense(tensor: torch.Tensor) -> np.ndarray:
    """Absolute values of a (possibly sparse) matrix as a dense array.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense or sparse matrix.

    Returns
    -------
    numpy.ndarray
        ``abs(tensor)`` as a dense float array.
    """
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    return np.abs(tensor.detach().cpu().numpy().astype(float))


def lifted_cells(data: Data) -> dict[str, list[tuple[int, ...]]]:
    """Read the higher-order cells of a lifted molecule as atom-index tuples.

    Hypergraphs give their hyperedges. For simplicial, cell and combinatorial
    complexes the atoms of every cell of rank 2 and higher are obtained by
    multiplying the incidence matrices down to the nodes. Rank 0 (atoms) and
    rank 1 (edges) are not returned, since the molecular graph already shows
    them.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The lifted data.

    Returns
    -------
    dict[str, list[tuple[int, ...]]]
        ``"hyperedges"`` for a hypergraph, else ``"rank 2"``, ``"rank 3"``, ...
        mapped to the atom indices of each cell.
    """
    if "incidence_hyperedges" in data:
        h = _dense(data["incidence_hyperedges"])
        return {
            "hyperedges": [
                tuple(np.flatnonzero(h[:, j]).tolist())
                for j in range(h.shape[1])
            ]
        }
    cells: dict[str, list[tuple[int, ...]]] = {}
    if "incidence_1" not in data:
        return cells
    members = _dense(data["incidence_1"]) > 0  # atoms of each edge
    rank = 2
    while f"incidence_{rank}" in data:
        boundary = _dense(data[f"incidence_{rank}"]) > 0
        if boundary.shape[0] != members.shape[1]:
            break
        members = (members.astype(float) @ boundary.astype(float)) > 0
        cells[f"rank {rank}"] = [
            tuple(np.flatnonzero(members[:, j]).tolist())
            for j in range(members.shape[1])
        ]
        rank += 1
    return cells


@contextmanager
def captured_combinatorial_cells() -> Iterator[
    list[tuple[int, tuple[int, ...]]]
]:
    """Record the cells a combinatorial lifting adds to its complex.

    The incidence matrices of a combinatorial complex cannot always give back
    the atoms of a cell: a rank-3 cell that contains no rank-2 cell, for
    example, is a column of zeros in ``incidence_3``. While this context is
    active, every ``CombinatorialComplex.add_cell`` call is recorded, so the
    atoms of the cells are known exactly as the lifting built them.

    Yields
    ------
    list[tuple[int, tuple[int, ...]]]
        ``(rank, sorted atom indices)`` of each cell added, filled while the
        context is active.
    """
    from toponetx.classes import CombinatorialComplex

    recorded: list[tuple[int, tuple[int, ...]]] = []
    original = CombinatorialComplex.add_cell

    def add_cell(self, cell, rank=None, *args, **kwargs):
        """Record the cell, then add it as usual (``self`` is the complex).

        Parameters
        ----------
        cell : Iterable
            Atoms of the cell.
        rank : int, optional
            Rank of the cell.
        *args : Any
            Passed on.
        **kwargs : Any
            Passed on.

        Returns
        -------
        Any
            Whatever ``CombinatorialComplex.add_cell`` returns.
        """
        with contextlib.suppress(TypeError, ValueError):
            recorded.append((int(rank), tuple(sorted(int(v) for v in cell))))
        return original(self, cell, rank, *args, **kwargs)

    CombinatorialComplex.add_cell = add_cell
    try:
        yield recorded
    finally:
        CombinatorialComplex.add_cell = original


def lift_to_cells(
    numbers: Any,
    positions: Any,
    liftings: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, list[tuple[int, ...]]]:
    """Lift one molecule and return its cells (see :func:`lifted_cells`).

    Parameters
    ----------
    numbers : array-like
        Atomic numbers.
    positions : array-like
        Positions in Angstrom.
    liftings : list[dict[str, Any]]
        Lifting configs as returned by :func:`model_liftings`.
    **kwargs : Any
        Passed to :func:`lift_molecule` (``graph_edges``, ``bond_scale``,
        ``cutoff``).

    Returns
    -------
    dict[str, list[tuple[int, ...]]]
        The cells of each rank.
    """
    with captured_combinatorial_cells() as recorded:
        data = lift_molecule(numbers, positions, liftings, **kwargs)
    cells = lifted_cells(data)
    # A cell is its set of atoms: adding the same set at another rank replaces
    # the first, so the last rank added is the one in the lifted complex.
    final = {atoms: rank for rank, atoms in recorded if rank >= 2}
    if final:
        # A combinatorial lifting: use the cells as it built them.
        cells = {}
        for atoms, rank in final.items():
            cells.setdefault(f"rank {rank}", []).append(atoms)
    return cells
