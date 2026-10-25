"""Parity checks for sparse k-hop lifting, including isolated atoms."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
)
from topobench.transforms.liftings.graph2hypergraph.khop_lifting_large_scale import (
    HypergraphKHopLiftingLargeScale,
)


@pytest.mark.parametrize("hops", [0, 1, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_matches_reference(hops, empty):
    edges = (
        torch.empty((2, 0), dtype=torch.long)
        if empty
        else torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    )
    data = Data(
        x=torch.arange(5, dtype=torch.float).view(-1, 1),
        edge_index=edges,
        num_nodes=5,
    )
    reference = HypergraphKHopLifting(k_value=hops)(data.clone())
    actual = HypergraphKHopLiftingLargeScale(k_value=hops)(data.clone())
    assert actual.num_hyperedges == reference.num_hyperedges == 5
    torch.testing.assert_close(
        actual.incidence_hyperedges.to_dense(),
        reference.incidence_hyperedges.to_dense(),
    )
    torch.testing.assert_close(actual.x_hyperedges, reference.x_hyperedges)
    assert actual.incidence_hyperedges.to_dense()[4, 4] == 1
