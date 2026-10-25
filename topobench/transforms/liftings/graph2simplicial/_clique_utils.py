"""Shared helpers for direct clique lifting implementations."""

from itertools import combinations

import igraph as ig
import networkx as nx
import numpy as np
import torch


def build_clique_complex_incidences(
    graph: nx.Graph,
    complex_dim: int,
    signed: bool,
    num_nodes: int | None = None,
):
    """Build clique simplices and sparse incidence matrices.

    Parameters
    ----------
    graph : nx.Graph
        Input graph.
    complex_dim : int
        Maximum simplicial rank.
    signed : bool
        Whether incidence values should carry orientation signs.
    num_nodes : int, optional
        Number of feature-bearing graph nodes. When provided, edges outside
        ``range(num_nodes)`` are ignored.

    Returns
    -------
    tuple
        Incidences, simplex tensors by rank, shape, and canonical edge list.
    """
    num_nodes = graph.number_of_nodes() if num_nodes is None else num_nodes
    valid_nodes = set(range(num_nodes))
    graph = graph.subgraph(valid_nodes).copy()
    simplices_by_rank = [None] * (complex_dim + 1)
    simplices_by_rank[0] = torch.arange(num_nodes).view(-1, 1)

    sorted_edges = sorted(tuple(sorted(edge)) for edge in graph.edges())
    if complex_dim >= 1:
        simplices_by_rank[1] = (
            torch.tensor(sorted_edges, dtype=torch.long)
            if sorted_edges
            else torch.empty((0, 2), dtype=torch.long)
        )

    if complex_dim == 2:
        clique_graph = ig.Graph(
            n=num_nodes,
            edges=sorted_edges,
            directed=False,
        )
        triangles = sorted(
            tuple(sorted(triangle))
            for triangle in clique_graph.list_triangles()
        )
        simplices_by_rank[2] = (
            torch.tensor(triangles, dtype=torch.long)
            if triangles
            else torch.empty((0, 3), dtype=torch.long)
        )
    elif complex_dim > 2:
        cliques = list(nx.find_cliques(graph))
        for rank in range(2, complex_dim + 1):
            simplices = set()
            for clique in cliques:
                clique = sorted(clique)
                if len(clique) >= rank + 1:
                    simplices.update(combinations(clique, rank + 1))

            simplices_by_rank[rank] = (
                torch.tensor(sorted(simplices), dtype=torch.long)
                if simplices
                else torch.empty((0, rank + 1), dtype=torch.long)
            )

    incidences = {
        0: torch.sparse_coo_tensor(
            torch.stack(
                [
                    torch.zeros(num_nodes, dtype=torch.long),
                    torch.arange(num_nodes, dtype=torch.long),
                ]
            ),
            torch.ones(num_nodes, dtype=torch.float),
            size=(1, num_nodes),
        ).coalesce()
    }

    for rank in range(1, complex_dim + 1):
        k_simplices = simplices_by_rank[rank]
        km1_simplices = simplices_by_rank[rank - 1]

        if k_simplices.shape[0] == 0:
            incidences[rank] = torch.sparse_coo_tensor(
                size=(km1_simplices.shape[0], 0)
            ).coalesce()
            continue

        num_k = k_simplices.shape[0]
        simplex_size = rank + 1
        all_faces = []

        for idx in range(simplex_size):
            mask = torch.ones(simplex_size, dtype=torch.bool)
            mask[idx] = False
            all_faces.append(k_simplices[:, mask])

        faces_tensor = torch.cat(all_faces, dim=0)
        row_indices = find_simplex_indices(
            km1_simplices.numpy(), faces_tensor.numpy()
        )
        col_indices = torch.arange(num_k).repeat(simplex_size)

        if signed:
            single_vals = torch.tensor(
                [(-1.0) ** idx for idx in range(simplex_size)]
            )
            vals = single_vals.repeat_interleave(num_k)
        else:
            vals = torch.ones(simplex_size * num_k)

        incidences[rank] = torch.sparse_coo_tensor(
            torch.stack(
                [torch.from_numpy(row_indices.astype(np.int64)), col_indices]
            ),
            vals,
            size=(km1_simplices.shape[0], k_simplices.shape[0]),
        ).coalesce()

    shape = [simplices.shape[0] for simplices in simplices_by_rank]
    return incidences, simplices_by_rank, shape, sorted_edges


def get_preserved_edge_features(graph: nx.Graph, sorted_edges):
    """Return edge features in canonical edge order if present.

    Parameters
    ----------
    graph : nx.Graph
        Input graph containing optional edge features.
    sorted_edges : list
        Canonically sorted undirected edges.

    Returns
    -------
    torch.Tensor or None
        Edge features in canonical edge order, or ``None`` when unavailable.
    """
    edge_features = []
    for edge in sorted_edges:
        edge_data = graph.get_edge_data(*edge) or {}
        if "features" not in edge_data:
            return None
        edge_features.append(edge_data["features"])
    return torch.stack(edge_features) if edge_features else None


def find_simplex_indices(target: np.ndarray, query: np.ndarray):
    """Find query rows in a lexicographically sorted target array.

    Parameters
    ----------
    target : np.ndarray
        Array containing known simplex rows.
    query : np.ndarray
        Array containing simplex rows to locate.

    Returns
    -------
    np.ndarray
        Row indices of each query simplex in ``target``.
    """
    target_map = {tuple(row): idx for idx, row in enumerate(target.tolist())}
    return np.asarray(
        [target_map[tuple(row)] for row in query.tolist()], dtype=np.int64
    )
