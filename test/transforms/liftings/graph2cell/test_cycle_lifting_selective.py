"""Parity checks for selective cycle lifting."""

import networkx as nx
import pytest
import torch
from topomodelx.nn.cell.cwn import CWN
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from topobench.nn.backbones.cell.cccn import CCCN
from topobench.transforms.liftings.graph2cell.cycle_lifting import (
    CellCycleLifting,
    CellCycleLiftingSelective,
)


@pytest.mark.parametrize("strategy", ["pairs", "sparse_mm"])
@pytest.mark.parametrize(
    "graph",
    [
        nx.cycle_graph(4),
        nx.complete_graph(5),
        nx.complete_bipartite_graph(2, 3),
        nx.path_graph(4),
        nx.empty_graph(4),
    ],
)
def test_matches_reference(graph, strategy):
    data = from_networkx(graph)
    data.x = torch.arange(graph.number_of_nodes(), dtype=torch.float).view(
        -1, 1
    )
    data.y = torch.tensor([1.25])
    data.pos = torch.zeros(graph.number_of_nodes(), 3)
    data.openqdc_index = torch.tensor([7])
    neighborhoods = [
        "up_adjacency-0",
        "up_adjacency-1",
        "down_adjacency-1",
        "down_adjacency-2",
        "up_laplacian-1",
        "down_laplacian-1",
        "up_incidence-0",
        "up_incidence-1",
        "down_incidence-1",
        "down_incidence-2",
        "2-up_adjacency-0",
    ]
    reference = CellCycleLifting(neighborhoods=neighborhoods)(data.clone())
    actual = CellCycleLiftingSelective(
        neighborhoods=neighborhoods, adjacency_strategy=strategy
    )(data.clone())
    for key in ["incidence_1", "incidence_2", *neighborhoods]:
        torch.testing.assert_close(
            actual[key].to_dense(), reference[key].to_dense()
        )
    for key in ["x_0", "x_1", "x_2", "y", "pos", "openqdc_index"]:
        torch.testing.assert_close(actual[key], reference[key])


@pytest.mark.parametrize("strategy", ["pairs", "sparse_mm"])
def test_overlapping_cycles_preserve_cwn_output(strategy):
    torch.manual_seed(42)
    data = from_networkx(nx.complete_bipartite_graph(2, 3))
    data.x = torch.randn(5, 3)
    neighborhoods = ["up_adjacency-1", "down_incidence-2"]
    reference = CellCycleLifting(neighborhoods=neighborhoods)(data.clone())
    actual = CellCycleLiftingSelective(
        neighborhoods=neighborhoods, adjacency_strategy=strategy
    )(data.clone())
    assert reference.adjacency_1.values().max() == 2
    model = CWN(3, 3, 3, hid_channels=8, n_layers=1).eval()
    with torch.no_grad():
        expected = model(
            reference.x_0,
            reference.x_1,
            reference.x_2,
            reference.adjacency_1,
            reference.incidence_2,
            reference.incidence_1.T,
        )
        output = model(
            actual.x_0,
            actual.x_1,
            actual.x_2,
            actual.adjacency_1,
            actual.incidence_2,
            actual.incidence_1.T,
        )
    for result, target in zip(output, expected, strict=True):
        torch.testing.assert_close(result, target)


def test_preserves_edge_features():
    data = Data(
        x=torch.ones(3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]]),
        edge_attr=torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]]),
    )
    kwargs = dict(preserve_edge_attr=True, neighborhoods=["up_adjacency-0"])
    reference = CellCycleLifting(**kwargs)(data.clone())
    actual = CellCycleLiftingSelective(**kwargs)(data.clone())
    torch.testing.assert_close(actual.x_1, reference.x_1)


def test_preserves_cycle_length_limit():
    data = from_networkx(nx.cycle_graph(10))
    data.x = torch.ones(10, 1)
    for limit, count in [(9, 0), (10, 1)]:
        result = CellCycleLiftingSelective(
            max_cell_length=limit, neighborhoods=[]
        )(data.clone())
        assert result.incidence_2.shape == (10, count)


def test_pre_normalization_preserves_cccn_output_and_gradients():
    torch.manual_seed(42)
    data = from_networkx(nx.complete_graph(4))
    data.x = torch.randn(4, 3)
    neighborhoods = ["down_laplacian-1", "up_laplacian-1"]
    reference = CellCycleLifting(neighborhoods=neighborhoods)(data.clone())
    actual = CellCycleLiftingSelective(
        neighborhoods=neighborhoods, normalize_laplacians=True
    )(data.clone())
    old_model = CCCN(3, n_layers=1)
    new_model = CCCN(3, n_layers=1, use_pre_normalized_operators=True)
    new_model.load_state_dict(old_model.state_dict())
    old_x = reference.x_1.detach().requires_grad_()
    new_x = actual.x_1.detach().requires_grad_()
    expected = old_model(
        old_x, reference.down_laplacian_1, reference.up_laplacian_1
    )
    output = new_model(new_x, actual.down_laplacian_1, actual.up_laplacian_1)
    torch.testing.assert_close(output, expected)
    expected.square().sum().backward()
    output.square().sum().backward()
    torch.testing.assert_close(new_x.grad, old_x.grad)
