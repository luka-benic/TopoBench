"""This module implements the cycle lifting for graphs to cell complexes."""

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import CellComplex

from topobench.data.utils.utils import (
    get_connectivity_from_incidences_selective,
)
from topobench.transforms.liftings.graph2cell.base import (
    Graph2CellLifting,
)


class CellCycleLifting(Graph2CellLifting):
    r"""Lift graphs to cell complexes.

    The algorithm creates 2-cells by identifying the cycles and considering them as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_cell_length=None, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.max_cell_length = max_cell_length

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Find the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        G = self._generate_graph_from_data(data)
        cycles = nx.cycle_basis(G)
        cell_complex = CellComplex(G)

        # Eliminate self-loop cycles
        cycles = [cycle for cycle in cycles if len(cycle) != 1]
        # Eliminate cycles that are greater than the max_cell_lenght
        if self.max_cell_length is not None:
            cycles = [
                cycle for cycle in cycles if len(cycle) <= self.max_cell_length
            ]
        cycles = sorted(cycles, key=lambda c: (len(c), tuple(sorted(c))))
        if len(cycles) != 0:
            cell_complex.add_cells_from(cycles, rank=self.complex_dim)
        return self._get_lifted_topology(cell_complex, G)


class CellCycleLiftingSelective(Graph2CellLifting):
    r"""Lift graphs to cell complexes with selective connectivity construction.

    The algorithm creates 2-cells by identifying cycles and uses direct sparse tensor
    construction. It avoids building a full TopoNetX CellComplex and computes only
    the selectively requested neighborhoods to prevent Out Of Memory (OOM) errors.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    adjacency_strategy : str, optional
        Strategy to build adjacency matrices ('sparse_mm' or 'pairs'). Default is 'pairs'.
    chunk_size : int, optional
        Chunk size for pair-based adjacency generation. Default is 1000.
    normalize_laplacians : bool, optional
        Whether to normalize Laplacian tensors. Default is False.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        max_cell_length=None,
        adjacency_strategy="pairs",
        chunk_size=1000,
        normalize_laplacians=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.max_cell_length = max_cell_length
        self.adjacency_strategy = adjacency_strategy
        self.chunk_size = chunk_size
        self.normalize_laplacians = normalize_laplacians

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Find cycles of a graph and construct selective connectivity.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        num_nodes = graph.number_of_nodes()

        # Canonical edge ordering
        sorted_edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
        edge_map = {e: i for i, e in enumerate(sorted_edges)}
        num_edges = len(sorted_edges)

        # Find cycles
        cycles = nx.cycle_basis(graph)

        # Filter
        cycles = [c for c in cycles if len(c) != 1]
        if self.max_cell_length is not None:
            cycles = [c for c in cycles if len(c) <= self.max_cell_length]
        cycles = sorted(cycles, key=lambda c: (len(c), tuple(sorted(c))))

        num_cells = len(cycles)

        incidences = {}

        # incidence_0
        incidences[0] = torch.sparse_coo_tensor(size=(0, num_nodes)).coalesce()

        # incidence_1
        edge_tensor = torch.tensor(sorted_edges, dtype=torch.long)
        num_k1 = edge_tensor.shape[0]
        if num_k1 > 0:
            rows_1 = torch.cat([edge_tensor[:, 0], edge_tensor[:, 1]])
            cols_1 = torch.cat([torch.arange(num_k1), torch.arange(num_k1)])
            vals_1 = torch.cat([-torch.ones(num_k1), torch.ones(num_k1)])
            incidences[1] = torch.sparse_coo_tensor(
                torch.stack([rows_1, cols_1]),
                vals_1,
                size=(num_nodes, num_edges),
            ).coalesce()
        else:
            incidences[1] = torch.sparse_coo_tensor(
                size=(num_nodes, num_edges)
            ).coalesce()

        # incidence_2 and cycle_edges
        cycle_edges = []
        if num_cells > 0:
            rows_2 = []
            cols_2 = []
            vals_2 = []

            for cell_id, cycle in enumerate(cycles):
                c_edges = []

                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]

                    key = (min(u, v), max(u, v))
                    edge_id = edge_map[key]

                    c_edges.append(edge_id)
                    rows_2.append(edge_id)
                    cols_2.append(cell_id)

                    sign = 1.0 if (u, v) == key else -1.0
                    vals_2.append(sign)

                cycle_edges.append(c_edges)

            incidences[2] = torch.sparse_coo_tensor(
                torch.stack(
                    [
                        torch.tensor(rows_2, dtype=torch.long),
                        torch.tensor(cols_2, dtype=torch.long),
                    ]
                ),
                torch.tensor(vals_2, dtype=torch.float),
                size=(num_edges, num_cells),
            ).coalesce()
        else:
            incidences[2] = torch.sparse_coo_tensor(
                size=(num_edges, 0)
            ).coalesce()

        # Connectivities
        shape = [num_nodes, num_edges, num_cells]
        lifted_topology = get_connectivity_from_incidences_selective(
            incidences=incidences,
            shape=shape,
            max_rank=self.complex_dim,
            neighborhoods=self.neighborhoods,
            signed=False,
            include_all_incidences=True,
            legacy_keys=True,
            cycles=cycles,
            cycle_edges=cycle_edges,
            adjacency_strategy=self.adjacency_strategy,
            chunk_size=self.chunk_size,
            normalize_laplacians=self.normalize_laplacians,
        )

        # Features
        lifted_topology["x_0"] = data.x

        if (
            self.contains_edge_attr
            and num_edges == graph.number_of_edges()
            and hasattr(data, "edge_index")
        ):
            ei = data.edge_index
            data_edge_map = {}
            for idx in range(ei.size(1)):
                u_n, v_n = ei[0, idx].item(), ei[1, idx].item()
                if u_n != v_n:
                    key = (min(u_n, v_n), max(u_n, v_n))
                    if key not in data_edge_map:
                        data_edge_map[key] = idx
            reorder_indices = []
            for e in sorted_edges:
                if e not in data_edge_map:
                    raise ValueError(
                        f"Edge {e} from NetworkX graph not found in edge_index."
                    )
                reorder_indices.append(data_edge_map[e])
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                lifted_topology["x_1"] = data.edge_attr[reorder_indices]
        return lifted_topology
