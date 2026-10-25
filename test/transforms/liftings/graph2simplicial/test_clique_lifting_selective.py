"""Tests for selective simplicial clique lifting."""

from pathlib import Path

import hydra
import pytest
import torch
import torch_geometric
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
    SimplicialCliqueLiftingSelective,
)
from topobench.utils.config_resolvers import register_all_resolvers

SCCNN_NEIGHBORHOODS = [
    "hodge_laplacian-0",
    "down_laplacian-1",
    "up_laplacian-1",
    "down_laplacian-2",
    "up_laplacian-2",
]
MATRIX_FREE_NEIGHBORHOODS = ["incidence-1", "incidence-2"]


def test_selective_matches_reference_for_sccnn_neighborhoods(simple_graph_1):
    """Compare selective lifting against the reference clique lifting.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Fixture graph used for lifting comparisons.
    """
    ref = SimplicialCliqueLifting(
        complex_dim=2, signed=False, neighborhoods=SCCNN_NEIGHBORHOODS
    ).forward(simple_graph_1.clone())
    selective = SimplicialCliqueLiftingSelective(
        complex_dim=2, signed=False, neighborhoods=SCCNN_NEIGHBORHOODS
    ).forward(simple_graph_1.clone())

    _assert_common_lifted_data(ref, selective, max_rank=2)
    for key in [
        "hodge_laplacian-0",
        "hodge_laplacian_0",
        "down_laplacian-1",
        "down_laplacian_1",
        "up_laplacian-1",
        "up_laplacian_1",
        "down_laplacian-2",
        "down_laplacian_2",
        "up_laplacian-2",
        "up_laplacian_2",
    ]:
        _assert_sparse_close(ref[key], selective[key])


def test_selective_matches_reference_for_signed_hodge(simple_graph_1):
    """Verify signed Hodge Laplacians through rank three.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Fixture graph used for lifting comparisons.
    """
    neighborhoods = [
        "hodge_laplacian-0",
        "hodge_laplacian-1",
        "hodge_laplacian-2",
        "hodge_laplacian-3",
    ]
    ref = SimplicialCliqueLifting(
        complex_dim=3, signed=True, neighborhoods=neighborhoods
    ).forward(simple_graph_1.clone())
    selective = SimplicialCliqueLiftingSelective(
        complex_dim=3, signed=True, neighborhoods=neighborhoods
    ).forward(simple_graph_1.clone())

    _assert_common_lifted_data(ref, selective, max_rank=3)
    for rank in range(4):
        _assert_sparse_close(
            ref[f"hodge_laplacian-{rank}"],
            selective[f"hodge_laplacian-{rank}"],
        )
        _assert_sparse_close(
            ref[f"hodge_laplacian_{rank}"],
            selective[f"hodge_laplacian_{rank}"],
        )


def test_selective_matches_reference_for_topotune_neighborhoods(
    simple_graph_1,
):
    """Verify multi-hop neighborhoods used by TopoTune configs.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Fixture graph used for lifting comparisons.
    """
    neighborhoods = [
        "up_adjacency-1",
        "up_incidence-0",
        "down_incidence-2",
        "2-up_adjacency-0",
    ]
    ref = SimplicialCliqueLifting(
        complex_dim=3, signed=False, neighborhoods=neighborhoods
    ).forward(simple_graph_1.clone())
    selective = SimplicialCliqueLiftingSelective(
        complex_dim=3, signed=False, neighborhoods=neighborhoods
    ).forward(simple_graph_1.clone())

    _assert_common_lifted_data(ref, selective, max_rank=3)
    for key in neighborhoods:
        _assert_sparse_close(ref[key], selective[key])


def test_selective_preserves_edge_features():
    """Verify edge attributes are preserved in canonical edge order."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    data = torch_geometric.data.Data(
        x=torch.arange(3, dtype=torch.float).view(-1, 1),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=3,
    )
    neighborhoods = ["hodge_laplacian-0"]

    ref = SimplicialCliqueLifting(
        complex_dim=2,
        signed=False,
        preserve_edge_attr=True,
        neighborhoods=neighborhoods,
    ).forward(data.clone())
    selective = SimplicialCliqueLiftingSelective(
        complex_dim=2,
        signed=False,
        preserve_edge_attr=True,
        neighborhoods=neighborhoods,
    ).forward(data.clone())

    _assert_common_lifted_data(ref, selective, max_rank=2)
    assert torch.allclose(
        selective.x_1,
        torch.tensor([[1.0], [3.0], [2.0]]),
    )


def test_selective_lists_each_triangle_once_in_two_dimensional_lifting():
    """Verify direct triangle listing on overlapping maximal cliques."""
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    edge_index = torch.tensor(
        [
            [node for edge in edges for node in (edge[0], edge[1])],
            [node for edge in edges for node in (edge[1], edge[0])],
        ],
        dtype=torch.long,
    )
    data = torch_geometric.data.Data(
        x=torch.randn(5, 3), edge_index=edge_index, num_nodes=5
    )
    neighborhoods = ["hodge_laplacian-0"]

    ref = SimplicialCliqueLifting(
        complex_dim=2, signed=False, neighborhoods=neighborhoods
    ).forward(data.clone())
    selective = SimplicialCliqueLiftingSelective(
        complex_dim=2, signed=False, neighborhoods=neighborhoods
    ).forward(data.clone())

    assert selective.shape == [5, 8, 5]
    _assert_common_lifted_data(ref, selective, max_rank=2)


@pytest.mark.parametrize(
    ("edge_index", "expected_shape"),
    [
        (torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long), [4, 3, 0]),
        (torch.empty((2, 0), dtype=torch.long), [4, 0, 0]),
        (torch.tensor([[0], [1]], dtype=torch.long), [4, 1, 0]),
    ],
)
def test_selective_handles_sparse_edge_cases(edge_index, expected_shape):
    """Verify trees, empty-edge graphs, and isolated nodes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Sparse graph edge index to lift.
    expected_shape : list
        Expected lifted simplex counts by rank.
    """
    data = torch_geometric.data.Data(
        x=torch.randn(4, 3), edge_index=edge_index, num_nodes=4
    )
    neighborhoods = ["hodge_laplacian-0", "up_laplacian-1"]

    ref = SimplicialCliqueLifting(
        complex_dim=2, signed=False, neighborhoods=neighborhoods
    ).forward(data.clone())
    selective = SimplicialCliqueLiftingSelective(
        complex_dim=2, signed=False, neighborhoods=neighborhoods
    ).forward(data.clone())

    assert selective.shape == expected_shape
    _assert_common_lifted_data(ref, selective, max_rank=2)
    for key in ["hodge_laplacian-0", "up_laplacian-1"]:
        _assert_sparse_close(ref[key], selective[key])


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        (
            "scn",
            MATRIX_FREE_NEIGHBORHOODS,
        ),
        (
            "sccn",
            [
                "hodge_laplacian-0",
                "hodge_laplacian-1",
                "hodge_laplacian-2",
            ],
        ),
        ("sccnn", SCCNN_NEIGHBORHOODS),
        ("sccnn_custom", MATRIX_FREE_NEIGHBORHOODS),
        ("san", ["up_laplacian-1", "down_laplacian-1"]),
    ],
)
def test_clique_selective_config_uses_top_level_neighborhoods(
    model_name, expected
):
    """Verify selective config reads model-level neighborhoods.

    Parameters
    ----------
    model_name : str
        Simplicial model config name.
    expected : list
        Expected top-level neighborhoods.
    """
    cfg = _compose_run_config(
        [
            "dataset=graph/cocitation_cora",
            f"model=simplicial/{model_name}",
            "transforms=liftings/graph2simplicial/clique_selective",
        ]
    )
    neighborhoods = OmegaConf.to_container(
        cfg.transforms.neighborhoods, resolve=True
    )
    backbone = OmegaConf.to_container(cfg.model.backbone, resolve=False)

    assert cfg.transforms.transform_name == (
        "SimplicialCliqueLiftingSelective"
    )
    assert neighborhoods == expected
    assert "neighborhoods" not in backbone


@pytest.mark.parametrize("model_name", ["topotune", "topotune_onehasse"])
def test_clique_selective_config_keeps_topotune_backbone_neighborhoods(
    model_name,
):
    """Verify TopoTune still uses backbone-level neighborhoods.

    Parameters
    ----------
    model_name : str
        TopoTune model config name.
    """
    cfg = _compose_run_config(
        [
            "dataset=graph/cocitation_cora",
            f"model=simplicial/{model_name}",
            "transforms=liftings/graph2simplicial/clique_selective",
        ]
    )
    transform_neighborhoods = OmegaConf.to_container(
        cfg.transforms.neighborhoods, resolve=True
    )
    backbone_neighborhoods = OmegaConf.to_container(
        cfg.model.backbone.neighborhoods, resolve=True
    )

    assert transform_neighborhoods == backbone_neighborhoods


def _assert_common_lifted_data(ref, selective, max_rank):
    """Compare shared lifted tensors and shapes.

    Parameters
    ----------
    ref : torch_geometric.data.Data
        Reference lifted data.
    selective : torch_geometric.data.Data
        Selectively lifted data.
    max_rank : int
        Highest rank to compare.
    """
    assert selective.shape == ref.shape
    for rank in range(max_rank + 1):
        _assert_sparse_close(
            ref[f"incidence_{rank}"], selective[f"incidence_{rank}"]
        )
        assert torch.allclose(ref[f"x_{rank}"], selective[f"x_{rank}"])


def _assert_sparse_close(left, right):
    """Compare sparse or dense tensors.

    Parameters
    ----------
    left : torch.Tensor
        First tensor to compare.
    right : torch.Tensor
        Second tensor to compare.
    """
    left = left.to_dense() if left.is_sparse else left
    right = right.to_dense() if right.is_sparse else right
    assert left.shape == right.shape
    assert torch.allclose(left, right)


def _compose_run_config(overrides):
    """Compose the main Hydra config for test assertions.

    Parameters
    ----------
    overrides : list
        Hydra overrides to apply.

    Returns
    -------
    omegaconf.DictConfig
        Composed Hydra configuration.
    """
    register_all_resolvers()
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=str(Path.cwd() / "configs")
    ):
        return hydra.compose(
            config_name="run.yaml",
            overrides=overrides,
            return_hydra_config=True,
        )
