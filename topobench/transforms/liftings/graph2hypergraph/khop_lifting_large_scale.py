"""Optimized k-hop lifting for large-scale graphs.

This module constructs k-hop reachability with sparse matrix operations.
Memory usage depends on the number of reachable node pairs.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from topobench.transforms.liftings.graph2hypergraph import (
    Graph2HypergraphLifting,
)


class HypergraphKHopLiftingLargeScale(Graph2HypergraphLifting):
    r"""Lift graph to hypergraph using k-hop neighborhoods (optimized).

    This is an optimized version of HypergraphKHopLifting that:
    - Uses sparse matrix operations throughout
    - Avoids Python loops over nodes
    - Is GPU-compatible

    Parameters
    ----------
    k_value : int, optional
        The number of hops to consider. Default is 1.
    **kwargs : optional
        Additional arguments for the class.

    Examples
    --------
    >>> lifting = HypergraphKHopLiftingLargeScale(k_value=2)
    >>> lifted = lifting(data)
    """

    def __init__(self, k_value: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k!r})"

    def lift_topology(self, data: Data) -> dict:
        r"""Lift graph to hypergraph using sparse k-hop computation.

        Each node's k-hop neighborhood becomes a hyperedge.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology with sparse incidence matrix.
        """
        num_nodes = data.num_nodes
        device = data.edge_index.device
        edge_index = to_undirected(data.edge_index)

        # Handle isolated nodes by adding self-loops
        all_nodes = torch.arange(num_nodes, device=device)
        nodes_with_edges = torch.unique(edge_index[0])
        isolated = all_nodes[~torch.isin(all_nodes, nodes_with_edges)]

        if isolated.numel() > 0:
            self_loops = torch.stack([isolated, isolated], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        # Build sparse adjacency matrix with self-loops for "include self"
        # Add self-loops so each node is in its own neighborhood
        self_loops_all = torch.stack([all_nodes, all_nodes], dim=0)
        edge_index_with_self = (
            self_loops_all
            if self.k == 0
            else torch.cat([edge_index, self_loops_all], dim=1)
        )

        # Create sparse adjacency
        values = torch.ones(edge_index_with_self.size(1), device=device)
        adj = torch.sparse_coo_tensor(
            edge_index_with_self,
            values,
            size=(num_nodes, num_nodes),
        ).coalesce()

        # Compute k-hop reachability via sparse matrix power
        # A^k[i,j] > 0 means j is reachable from i in k hops
        khop_adj = adj
        for _ in range(self.k - 1):
            khop_adj = torch.sparse.mm(khop_adj, adj)
            # Binarize to avoid numerical growth
            khop_adj = khop_adj.coalesce()
            indices = khop_adj.indices()
            values = torch.ones(indices.size(1), device=device)
            khop_adj = torch.sparse_coo_tensor(
                indices, values, size=(num_nodes, num_nodes)
            ).coalesce()

        # The k-hop adjacency matrix IS our incidence matrix!
        # Row i = hyperedge for node i
        # We need to transpose: incidence[node, hyperedge]
        # Currently khop_adj[node, neighbor] = 1 means neighbor in node's khop
        # We want incidence[neighbor, node] = 1 (node i's hyperedge contains neighbor)

        # Transpose: incidence_hyperedges[j, i] = 1 means node j is in hyperedge i
        incidence = khop_adj.t().coalesce()

        # Binarize to ensure matching standard binary representation
        indices = incidence.indices()
        values = torch.ones(indices.size(1), device=device)
        incidence = torch.sparse_coo_tensor(
            indices, values, size=(num_nodes, num_nodes)
        ).coalesce()

        return {
            "incidence_hyperedges": incidence,
            "num_hyperedges": num_nodes,
            "x_0": data.x,
        }
