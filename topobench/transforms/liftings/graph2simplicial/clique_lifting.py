"""This module implements the CliqueLifting class, which lifts graphs to simplicial complexes."""

from itertools import combinations
from typing import Any

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobench.data.utils.utils import (
    get_simplicial_connectivity_from_incidences_selective,
)
from topobench.transforms.liftings.graph2simplicial import (
    Graph2SimplicialLifting,
)
from topobench.transforms.liftings.graph2simplicial._clique_utils import (
    build_clique_complex_incidences,
    get_preserved_edge_features,
)


class SimplicialCliqueLifting(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complex domain.

    The algorithm creates simplices by identifying the cliques and considering them as simplices of the same dimension.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to a simplicial complex.

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
        simplicial_complex = SimplicialComplex(graph)
        cliques = nx.find_cliques(graph)
        simplices: list[set[tuple[Any, ...]]] = [
            set() for _ in range(2, self.complex_dim + 1)
        ]
        for clique in cliques:
            for i in range(2, self.complex_dim + 1):
                for c in combinations(clique, i + 1):
                    simplices[i - 2].add(tuple(c))

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        return self._get_lifted_topology(simplicial_complex, graph)


class SimplicialCliqueLiftingSelective(Graph2SimplicialLifting):
    r"""Lift graphs to simplicial complexes with selective connectivity.

    The algorithm identifies clique simplices, builds sparse incidence
    matrices directly, and constructs only the requested neighborhoods.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to a selective simplicial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The selectively lifted topology.
        """
        graph = self._generate_graph_from_data(data)
        incidences, simplices_by_rank, shape, sorted_edges = (
            build_clique_complex_incidences(
                graph=graph,
                complex_dim=self.complex_dim,
                signed=True,
                num_nodes=data.x.shape[0],
            )
        )

        lifted_topology = (
            get_simplicial_connectivity_from_incidences_selective(
                incidences=incidences,
                shape=shape,
                max_rank=self.complex_dim,
                neighborhoods=self.neighborhoods,
                signed=self.signed,
            )
        )
        lifted_topology["x_0"] = data.x

        if (
            self.contains_edge_attr
            and len(simplices_by_rank) > 1
            and simplices_by_rank[1].shape[0] == graph.number_of_edges()
        ):
            edge_features = get_preserved_edge_features(graph, sorted_edges)
            if edge_features is not None:
                lifted_topology["x_1"] = edge_features

        return lifted_topology
