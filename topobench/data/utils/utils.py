"""Data utilities."""

import hashlib
import warnings

import networkx as nx
import numpy as np
import omegaconf
import torch
import torch_geometric
import torch_geometric.utils
from topomodelx.utils.sparse import from_sparse
from toponetx.classes import SimplicialComplex


def get_routes_from_neighborhoods(neighborhoods):
    """Get the routes from the neighborhoods.

    Combination of src_rank, dst_rank. ex: [[0, 0], [1, 0], [1, 1], [1, 1], [2, 1]].

    Parameters
    ----------
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    list
        List of routes.
    """
    routes = []
    for neighborhood in neighborhoods:
        split = neighborhood.split("-")
        src_rank = int(split[-1])
        r = int(split[0]) if len(split) == 3 else 1

        # Adjacency neighborhoods are always intrarank (same src and dst rank)
        # since adjacency defines connections within the same rank
        if "adjacency" in neighborhood:
            route = [src_rank, src_rank]
        else:
            route = (
                [src_rank, src_rank - r]
                if "down" in neighborhood
                else [src_rank, src_rank + r]
            )
        routes.append(route)
    return routes


def get_complex_connectivity(
    complex, max_rank, neighborhoods=None, signed=False
):
    """Get the connectivity matrices for the complex.

    Parameters
    ----------
    complex : toponetx.CellComplex or toponetx.SimplicialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.
    signed : bool, optional
        If True, returns signed connectivity matrices.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]:
            try:  #### from_sparse doesn't have rank and signed
                connectivity[f"{connectivity_info}_{rank_idx}"] = from_sparse(
                    getattr(complex, f"{connectivity_info}_matrix")(
                        rank=rank_idx, signed=signed
                    )
                )
            except:  # noqa: E722
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    if neighborhoods is not None:
        connectivity = select_neighborhoods_of_interest(
            connectivity, neighborhoods
        )
    connectivity["shape"] = practical_shape
    return connectivity


def _parse_neighborhood_target(target):
    """Parse a connectivity target into type, rank, and hop distance.

    Parameters
    ----------
    target : str
        Neighborhood or canonical connectivity key.

    Returns
    -------
    tuple
        Parsed neighborhood type, rank, and hop distance.
    """
    if target.startswith("incidence_") and "-" not in target:
        return "incidence", int(target.rsplit("_", 1)[1]), 1

    if "-" in target:
        target_prefix, rank = target.rsplit("-", 1)
        if "-" in target_prefix:
            hop, neighborhood_type = target_prefix.split("-", 1)
            if hop.isdigit():
                return neighborhood_type.replace("-", "_"), int(rank), int(hop)
        neighborhood_type = target_prefix.replace("-", "_")
        if neighborhood_type == "adjacency":
            neighborhood_type = "up_adjacency"
        elif neighborhood_type == "coadjacency":
            neighborhood_type = "down_adjacency"
        return neighborhood_type, int(rank), 1

    target_type, rank = target.rsplit("_", 1)
    if target_type == "adjacency":
        target_type = "up_adjacency"
    elif target_type == "coadjacency":
        target_type = "down_adjacency"
    return target_type, int(rank), 1


def _connectivity_aliases(neighborhood_type, rank, hop=1):
    """Return compatible key aliases for a parsed connectivity target.

    Parameters
    ----------
    neighborhood_type : str
        Parsed neighborhood type.
    rank : int
        Cell rank for the neighborhood.
    hop : int, optional
        Hop distance for multi-hop neighborhoods.

    Returns
    -------
    set
        Compatible aliases for the neighborhood.
    """
    hyphen_type = neighborhood_type.replace("_", "-")
    aliases = {
        f"{neighborhood_type}-{rank}",
        f"{hyphen_type}-{rank}",
    }

    if hop != 1:
        return {
            f"{hop}-{neighborhood_type}-{rank}",
            f"{hop}-{hyphen_type}-{rank}",
        }

    if neighborhood_type == "up_adjacency":
        aliases.add(f"adjacency_{rank}")
    elif neighborhood_type == "down_adjacency":
        aliases.add(f"coadjacency_{rank}")
    elif neighborhood_type in {
        "up_laplacian",
        "down_laplacian",
        "hodge_laplacian",
        "incidence",
    }:
        aliases.add(f"{neighborhood_type}_{rank}")

    return aliases


def _store_with_aliases(
    connectivity, target, value, neighborhood_type=None, rank=None, hop=1
):
    """Store a connectivity tensor under requested and canonical aliases.

    Parameters
    ----------
    connectivity : dict
        Connectivity dictionary to update.
    target : str
        Requested neighborhood key.
    value : torch.Tensor
        Connectivity tensor to store.
    neighborhood_type : str, optional
        Parsed neighborhood type.
    rank : int, optional
        Cell rank for the neighborhood.
    hop : int, optional
        Hop distance for multi-hop neighborhoods.
    """
    connectivity[target] = value
    if neighborhood_type is None or rank is None:
        neighborhood_type, rank, hop = _parse_neighborhood_target(target)
    for alias in _connectivity_aliases(neighborhood_type, rank, hop):
        connectivity.setdefault(alias, value)


def get_simplicial_connectivity_from_incidences_selective(
    incidences,
    shape,
    max_rank,
    neighborhoods=None,
    signed=False,
    required_keys=None,
):
    """Compute only requested simplicial connectivity matrices.

    Parameters
    ----------
    incidences : dict
        Dictionary mapping rank to sparse PyTorch incidence matrices.
    shape : list
        List of simplex counts per rank.
    max_rank : int
        Maximum simplex rank.
    neighborhoods : list, optional
        Neighborhood keys to construct.
    signed : bool, optional
        Whether to expose signed incidences and signed operators.
    required_keys : list, optional
        Extra connectivity keys to construct.

    Returns
    -------
    dict
        Dictionary containing incidences, requested connectivities, aliases,
        and shape.
    """
    practical_shape = list(
        np.pad(shape, (0, max(0, max_rank + 1 - len(shape))))
    )
    connectivity = {"shape": practical_shape}

    first_incidence = next(iter(incidences.values()), None)
    device = first_incidence.device if first_incidence is not None else None
    dtype = (
        first_incidence.dtype if first_incidence is not None else torch.float
    )

    def zero(m, n):
        """Create a zero sparse matrix on the incidence device.

        Parameters
        ----------
        m : int
            Number of rows.
        n : int
            Number of columns.

        Returns
        -------
        torch.Tensor
            Sparse zero matrix.
        """
        return zero_sparse(m, n, device=device, dtype=dtype)

    def raw_incidence(rank):
        """Return the signed incidence matrix for a rank.

        Parameters
        ----------
        rank : int
            Incidence rank to retrieve.

        Returns
        -------
        torch.Tensor
            Sparse incidence matrix.
        """
        if rank in incidences:
            return incidences[rank].coalesce()
        if rank <= 0:
            n_cols = practical_shape[0] if practical_shape else 0
            return zero(1, n_cols)
        if rank > max_rank:
            n_rows = (
                practical_shape[rank - 1]
                if rank - 1 < len(practical_shape)
                else 0
            )
            return zero(n_rows, 0)
        return zero(practical_shape[rank - 1], practical_shape[rank])

    def output_incidence(rank):
        """Return the externally exposed incidence matrix.

        Parameters
        ----------
        rank : int
            Incidence rank to retrieve.

        Returns
        -------
        torch.Tensor
            Sparse incidence matrix, signed only when requested.
        """
        incidence = raw_incidence(rank)
        if signed:
            return incidence
        return torch.sparse_coo_tensor(
            incidence.indices(),
            incidence.values().abs(),
            incidence.size(),
            device=incidence.device,
        ).coalesce()

    for rank_idx in range(max_rank + 1):
        connectivity[f"incidence_{rank_idx}"] = output_incidence(rank_idx)

    if neighborhoods is None and required_keys is None:
        warnings.warn(
            "No neighborhoods specified for selective builder. Returning only incidences to avoid OOM.",
            stacklevel=2,
        )
        neighborhoods = []

    targets = set(neighborhoods or []) | set(required_keys or [])
    product_cache = {}

    def processed_incidence(rank):
        """Return an incidence matrix for unsigned products.

        Parameters
        ----------
        rank : int
            Incidence rank to retrieve.

        Returns
        -------
        torch.Tensor
            Sparse incidence matrix used for exposed operators.
        """
        return output_incidence(rank) if not signed else raw_incidence(rank)

    def grouped_gram(incidence, layout):
        """Assemble a sparse Gram matrix grouped by simplex membership.

        Parameters
        ----------
        incidence : torch.Tensor
            Sparse incidence matrix.
        layout : str
            Product layout, either ``"up"`` or ``"down"``.

        Returns
        -------
        torch.Tensor
            Sparse Gram matrix.
        """
        incidence = incidence.coalesce()
        if incidence._nnz() == 0:
            rows, cols = incidence.size()
            return zero(rows, rows) if layout == "up" else zero(cols, cols)

        indices = incidence.indices()
        values = incidence.values()
        group_indices = indices[1] if layout == "up" else indices[0]
        item_indices = indices[0] if layout == "up" else indices[1]
        output_size = (
            incidence.size(0) if layout == "up" else incidence.size(1)
        )

        order = torch.argsort(group_indices)
        group_indices = group_indices[order]
        item_indices = item_indices[order]
        values = values[order]

        unique_groups, counts = torch.unique_consecutive(
            group_indices, return_counts=True
        )
        del unique_groups

        row_chunks = []
        col_chunks = []
        val_chunks = []
        start = 0
        for count in counts.tolist():
            end = start + count
            items = item_indices[start:end]
            group_values = values[start:end]
            row_chunks.append(items.repeat_interleave(count))
            col_chunks.append(items.repeat(count))
            val_chunks.append(
                group_values.repeat_interleave(count)
                * group_values.repeat(count)
            )
            start = end

        rows = torch.cat(row_chunks)
        cols = torch.cat(col_chunks)
        vals = torch.cat(val_chunks)
        return torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            size=(output_size, output_size),
            device=incidence.device,
        ).coalesce()

    def product(kind, rank, use_signed=False):
        """Return a cached one-step up or down product.

        Parameters
        ----------
        kind : str
            Product kind, either ``"up"`` or ``"down"``.
        rank : int
            Cell rank for the product.
        use_signed : bool, optional
            Whether to use signed incidences.

        Returns
        -------
        torch.Tensor
            Sparse up or down product.
        """
        cache_key = (kind, rank, use_signed)
        if cache_key in product_cache:
            return product_cache[cache_key]

        n_rank = practical_shape[rank]
        if kind == "up":
            incidence = (
                raw_incidence(rank + 1)
                if use_signed
                else processed_incidence(rank + 1)
            )
            result = (
                grouped_gram(incidence, layout="up")
                if incidence.size(1) != 0
                else zero(n_rank, n_rank)
            )
        elif kind == "down":
            incidence = (
                raw_incidence(rank)
                if use_signed
                else processed_incidence(rank)
            )
            result = (
                grouped_gram(incidence, layout="down")
                if rank >= 1 and incidence.size(0) != 0
                else zero(n_rank, n_rank)
            )
        else:
            raise ValueError(f"Invalid product kind: {kind}")

        product_cache[cache_key] = result
        return result

    def hodge_laplacian(rank):
        """Return the Hodge Laplacian for a rank.

        Parameters
        ----------
        rank : int
            Cell rank for the Hodge Laplacian.

        Returns
        -------
        torch.Tensor
            Sparse Hodge Laplacian.
        """
        hodge = (
            product("down", rank, use_signed=True)
            + product("up", rank, use_signed=True)
        ).coalesce()
        return hodge if signed else abs_sparse(hodge)

    def off_diagonal(matrix):
        """Return the off-diagonal entries of a sparse matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Sparse matrix to filter.

        Returns
        -------
        torch.Tensor
            Sparse matrix without diagonal entries.
        """
        matrix = matrix.coalesce()
        idx = matrix.indices()
        vals = matrix.values()
        mask = idx[0] != idx[1]
        if not signed:
            vals = vals.abs()
        return torch.sparse_coo_tensor(
            idx[:, mask], vals[mask], matrix.size(), device=matrix.device
        ).coalesce()

    def incidence_chain(src_rank, hop, direction):
        """Multiply consecutive incidence matrices along one direction.

        Parameters
        ----------
        src_rank : int
            Source rank for the requested neighborhood.
        hop : int
            Hop distance.
        direction : str
            Direction, either ``"up"`` or ``"down"``.

        Returns
        -------
        torch.Tensor
            Sparse multi-hop incidence product.
        """
        if direction == "up":
            matrix = output_incidence(src_rank + 1)
            for idx in range(src_rank + 2, src_rank + hop + 1):
                matrix = safe_sparse_mm(matrix, output_incidence(idx))
            return matrix

        matrix = output_incidence(src_rank - hop + 1)
        for idx in range(src_rank - hop + 2, src_rank + 1):
            matrix = safe_sparse_mm(matrix, output_incidence(idx))
        return matrix

    def binarized(matrix):
        """Return a sparse matrix with all non-zero values set to one.

        Parameters
        ----------
        matrix : torch.Tensor
            Sparse matrix to binarize.

        Returns
        -------
        torch.Tensor
            Binarized sparse matrix.
        """
        matrix = matrix.coalesce()
        if matrix._nnz() == 0:
            return matrix
        return torch.sparse_coo_tensor(
            matrix.indices(),
            torch.ones_like(matrix.values()),
            matrix.size(),
            device=matrix.device,
        ).coalesce()

    for target in targets:
        neighborhood_type, rank, hop = _parse_neighborhood_target(target)

        if neighborhood_type == "incidence":
            value = output_incidence(rank)
        elif neighborhood_type == "up_laplacian" and hop == 1:
            value = product("up", rank)
        elif neighborhood_type == "down_laplacian" and hop == 1:
            value = product("down", rank)
        elif neighborhood_type == "hodge_laplacian" and hop == 1:
            value = hodge_laplacian(rank)
        elif neighborhood_type == "up_adjacency" and hop == 1:
            value = off_diagonal(product("up", rank))
        elif neighborhood_type == "down_adjacency" and hop == 1:
            value = off_diagonal(product("down", rank))
        elif neighborhood_type in {"up_incidence", "down_incidence"}:
            direction = neighborhood_type.split("_", 1)[0]
            value = incidence_chain(rank, hop, direction)
            value = (
                binarized(value).T if direction == "up" else binarized(value)
            )
        elif neighborhood_type in {
            "up_laplacian",
            "down_laplacian",
            "up_adjacency",
            "down_adjacency",
        }:
            direction, operator_type = neighborhood_type.split("_", 1)
            matrix = incidence_chain(rank, hop, direction)
            matrix = (
                safe_sparse_mm(matrix, matrix.T)
                if direction == "up"
                else safe_sparse_mm(matrix.T, matrix)
            )
            matrix = binarized(matrix)
            value = (
                remove_diag_sparse(matrix)
                if operator_type == "adjacency"
                else matrix
            )
        else:
            raise ValueError(f"Unsupported neighborhood: {target}")

        _store_with_aliases(
            connectivity,
            target,
            value,
            neighborhood_type=neighborhood_type,
            rank=rank,
            hop=hop,
        )

    return connectivity


def zero_sparse(m, n, device=None, dtype=torch.float):
    """Generate a zero sparse tensor of given dimensions.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    device : torch.device, optional
        Device to store the tensor. Default is None.
    dtype : torch.dtype, optional
        Data type of the tensor. Default is torch.float.

    Returns
    -------
    torch.Tensor
        Zero sparse tensor.
    """
    return torch.sparse_coo_tensor(
        size=(m, n), device=device, dtype=dtype
    ).coalesce()


def binarize_sparse(A):
    """Binarize a sparse tensor by turning all non-zero values into 1.0.

    Parameters
    ----------
    A : torch.Tensor
        Sparse tensor to binarize.

    Returns
    -------
    torch.Tensor
        Binarized sparse tensor.
    """
    if not A.is_sparse:
        return A
    A = A.coalesce()
    vals = torch.ones_like(A.values())
    return torch.sparse_coo_tensor(
        A.indices(), vals, A.size(), device=A.device
    ).coalesce()


def abs_sparse(A):
    """Take the absolute value of all entries in a sparse tensor.

    Parameters
    ----------
    A : torch.Tensor
        Sparse tensor.

    Returns
    -------
    torch.Tensor
        Absolute sparse tensor.
    """
    if not A.is_sparse:
        return A
    A = A.coalesce()
    vals = torch.abs(A.values())
    return torch.sparse_coo_tensor(
        A.indices(), vals, A.size(), device=A.device
    ).coalesce()


def remove_diag_sparse(A, binarize=True):
    """Remove diagonal elements from a sparse tensor and optionally binarize.

    Parameters
    ----------
    A : torch.Tensor
        Sparse tensor.
    binarize : bool, optional
        Whether to binarize non-zero entries. Default is True.

    Returns
    -------
    torch.Tensor
        Sparse tensor with diagonal removed.
    """
    if not A.is_sparse:
        return A
    A = A.coalesce()
    indices = A.indices()
    values = A.values()
    mask = indices[0] != indices[1]
    new_indices = indices[:, mask]
    new_values = values[mask]
    if binarize:
        new_values = torch.ones_like(new_values)
    return torch.sparse_coo_tensor(
        new_indices, new_values, A.size(), device=A.device
    ).coalesce()


def safe_sparse_mm(A, B):
    """Safe sparse matrix multiplication.

    Parameters
    ----------
    A : torch.Tensor
        First sparse tensor.
    B : torch.Tensor
        Second sparse tensor.

    Returns
    -------
    torch.Tensor
        Product of A and B.
    """
    A = A.coalesce()
    B = B.coalesce()
    if A._nnz() == 0 or B._nnz() == 0:
        return zero_sparse(
            A.size(0), B.size(1), device=A.device, dtype=A.dtype
        )
    return torch.sparse.mm(A, B)


def build_edge_cycle_adjacency_from_cycle_edges(
    cycle_edges, num_edges, device=None, dtype=torch.float, chunk_size=1000
):
    """Return sparse COO equivalent to offdiag(abs(B2) @ abs(B2).T).

    Parameters
    ----------
    cycle_edges : list of lists
        List of edges in each cycle.
    num_edges : int
        Number of edges in the graph.
    device : torch.device, optional
        Device to store the tensor. Default is None.
    dtype : torch.dtype, optional
        Data type of the tensor. Default is torch.float.
    chunk_size : int, optional
        Chunk size for sparse tensor construction. Default is 1000.

    Returns
    -------
    torch.Tensor
        Sparse COO adjacency tensor.
    """
    if len(cycle_edges) == 0:
        return zero_sparse(num_edges, num_edges, device=device, dtype=dtype)

    all_rows, all_cols = [], []
    rows, cols = [], []

    for i, edges in enumerate(cycle_edges):
        if len(edges) < 2:
            continue
        e = torch.tensor(edges, dtype=torch.long, device=device)
        r = e.repeat_interleave(len(e))
        c = e.repeat(len(e))
        mask = r != c
        rows.append(r[mask])
        cols.append(c[mask])

        if chunk_size and (i + 1) % chunk_size == 0 and rows:
            all_rows.append(torch.cat(rows))
            all_cols.append(torch.cat(cols))
            rows, cols = [], []

    if rows:
        all_rows.append(torch.cat(rows))
        all_cols.append(torch.cat(cols))

    if not all_rows:
        return zero_sparse(num_edges, num_edges, device=device, dtype=dtype)

    row_t = torch.cat(all_rows)
    col_t = torch.cat(all_cols)
    val_t = torch.ones(len(row_t), dtype=dtype, device=device)

    adj = torch.sparse_coo_tensor(
        torch.stack([row_t, col_t]),
        val_t,
        size=(num_edges, num_edges),
        device=device,
    ).coalesce()
    return adj


def gcn_normalize_sparse_tensor(A: torch.Tensor) -> torch.Tensor:
    """Normalize a sparse COO tensor like GCNConv without adding self-loops.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO tensor to normalize.

    Returns
    -------
    torch.Tensor
        Normalized sparse COO tensor.
    """
    if A.layout != torch.sparse_coo:
        raise ValueError(f"Expected torch.sparse_coo, got {A.layout}")

    A = A.coalesce()
    if A._nnz() == 0:
        return A

    idx = A.indices()
    vals = A.values()

    row, col = idx[0], idx[1]

    # Compute degree
    deg = torch.zeros(A.size(0), dtype=vals.dtype, device=A.device)
    deg.scatter_add_(0, row, vals)

    # PyG degree formula: deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

    # Normalize values
    norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        idx, norm_vals, A.size(), device=A.device
    ).coalesce()


def build_edge_cycle_laplacian_from_cycle_edges(
    cycle_edges, num_edges, device=None, dtype=torch.float, chunk_size=1000
):
    """Return sparse COO equivalent to unsigned B2 @ B2.T.

    Parameters
    ----------
    cycle_edges : list of lists
        List of edges in each cycle.
    num_edges : int
        Number of edges in the graph.
    device : torch.device, optional
        Device to store the tensor. Default is None.
    dtype : torch.dtype, optional
        Data type of the tensor. Default is torch.float.
    chunk_size : int, optional
        Chunk size for sparse tensor construction. Default is 1000.

    Returns
    -------
    torch.Tensor
        Sparse COO Laplacian tensor.
    """
    if len(cycle_edges) == 0:
        return zero_sparse(num_edges, num_edges, device=device, dtype=dtype)

    all_rows, all_cols = [], []
    rows, cols = [], []

    for i, edges in enumerate(cycle_edges):
        if len(edges) == 0:
            continue

        e = torch.tensor(edges, dtype=torch.long, device=device)
        r = e.repeat_interleave(len(e))
        c = e.repeat(len(e))

        rows.append(r)
        cols.append(c)

        if chunk_size and (i + 1) % chunk_size == 0 and rows:
            all_rows.append(torch.cat(rows))
            all_cols.append(torch.cat(cols))
            rows, cols = [], []

    if rows:
        all_rows.append(torch.cat(rows))
        all_cols.append(torch.cat(cols))

    if not all_rows:
        return zero_sparse(num_edges, num_edges, device=device, dtype=dtype)

    row_t = torch.cat(all_rows)
    col_t = torch.cat(all_cols)
    val_t = torch.ones(len(row_t), dtype=dtype, device=device)

    L = torch.sparse_coo_tensor(
        torch.stack([row_t, col_t]),
        val_t,
        size=(num_edges, num_edges),
        device=device,
    ).coalesce()
    return L


def build_node_cycle_adjacency_from_cycles(
    cycles, num_nodes, device=None, dtype=torch.float, chunk_size=1000
):
    """Return sparse COO equivalent to offdiag((B1 @ B2) @ (B1 @ B2).T).

    Parameters
    ----------
    cycles : list of lists
        List of nodes in each cycle.
    num_nodes : int
        Number of nodes in the graph.
    device : torch.device, optional
        Device to store the tensor. Default is None.
    dtype : torch.dtype, optional
        Data type of the tensor. Default is torch.float.
    chunk_size : int, optional
        Chunk size for sparse tensor construction. Default is 1000.

    Returns
    -------
    torch.Tensor
        Sparse COO adjacency tensor.
    """
    if len(cycles) == 0:
        return zero_sparse(num_nodes, num_nodes, device=device, dtype=dtype)

    all_rows, all_cols = [], []
    rows, cols = [], []

    for i, nodes in enumerate(cycles):
        if len(nodes) < 2:
            continue
        n = torch.tensor(nodes, dtype=torch.long, device=device)
        r = n.repeat_interleave(len(n))
        c = n.repeat(len(n))
        mask = r != c
        rows.append(r[mask])
        cols.append(c[mask])

        if chunk_size and (i + 1) % chunk_size == 0 and rows:
            all_rows.append(torch.cat(rows))
            all_cols.append(torch.cat(cols))
            rows, cols = [], []

    if rows:
        all_rows.append(torch.cat(rows))
        all_cols.append(torch.cat(cols))

    if not all_rows:
        return zero_sparse(num_nodes, num_nodes, device=device, dtype=dtype)

    row_t = torch.cat(all_rows)
    col_t = torch.cat(all_cols)
    val_t = torch.ones(len(row_t), dtype=dtype, device=device)

    adj = torch.sparse_coo_tensor(
        torch.stack([row_t, col_t]),
        val_t,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()
    return binarize_sparse(adj)


def get_connectivity_from_incidences_selective(
    incidences,
    shape,
    max_rank,
    neighborhoods=None,
    signed=False,
    include_all_incidences=True,
    legacy_keys=True,
    cycles=None,
    cycle_edges=None,
    adjacency_strategy="pairs",
    required_keys=None,
    chunk_size=1000,
    normalize_laplacians=False,
):
    """Memory-aware selective connectivity builder.

    If neighborhoods is None, it warns and returns only incidences to prevent OOM.

    Parameters
    ----------
    incidences : dict
        Dictionary of incidence matrices.
    shape : list
        Shape of elements at each rank.
    max_rank : int
        Maximum rank to process.
    neighborhoods : list, optional
        List of neighborhoods to compute. Default is None.
    signed : bool, optional
        Whether to compute signed connectivity. Default is False.
    include_all_incidences : bool, optional
        Whether to include all incidence matrices in output. Default is True.
    legacy_keys : bool, optional
        Whether to use legacy key names. Default is True.
    cycles : list of lists, optional
        List of nodes in each cycle. Default is None.
    cycle_edges : list of lists, optional
        List of edges in each cycle. Default is None.
    adjacency_strategy : str, optional
        Adjacency computation strategy. Default is "pairs".
    required_keys : list, optional
        List of required keys. Default is None.
    chunk_size : int, optional
        Chunk size for sparse tensor construction. Default is 1000.
    normalize_laplacians : bool, optional
        Whether to normalize Laplacian tensors. Default is False.

    Returns
    -------
    dict
        Dictionary containing the constructed connectivity.
    """
    connectivity = {"shape": shape}

    if include_all_incidences:
        for r, inc in incidences.items():
            connectivity[f"incidence_{r}"] = inc if signed else abs_sparse(inc)

    if neighborhoods is None and required_keys is None:
        warnings.warn(
            "No neighborhoods specified for selective builder. Returning only incidences to avoid OOM.",
            stacklevel=2,
        )
        neighborhoods = []

    targets = set(neighborhoods or []) | set(required_keys or [])

    def get_inc(r):
        """Get the incidence matrix at rank r.

        Parameters
        ----------
        r : int
            Rank index.

        Returns
        -------
        torch.Tensor
            Incidence matrix at rank r.
        """
        if r in incidences:
            return incidences[r]
        if r <= 0:
            return zero_sparse(0, shape[0] if len(shape) > 0 else 0)
        if r > len(shape) - 1:
            return zero_sparse(shape[-1] if len(shape) > 0 else 0, 0)
        return zero_sparse(shape[r - 1], shape[r])

    for t in targets:
        t_normalized = t.replace("_", "-")

        if t_normalized == "up-adjacency-1" or t == "adjacency_1":
            if adjacency_strategy == "pairs" and cycle_edges is not None:
                ref_inc = get_inc(2)
                adj = build_edge_cycle_adjacency_from_cycle_edges(
                    cycle_edges,
                    shape[1],
                    device=ref_inc.device,
                    dtype=ref_inc.dtype,
                    chunk_size=chunk_size,
                )
            else:
                inc2 = abs_sparse(get_inc(2))
                adj = remove_diag_sparse(
                    safe_sparse_mm(inc2, inc2.T), binarize=False
                )
            connectivity[t] = adj
            if legacy_keys and t_normalized == "up-adjacency-1":
                connectivity["adjacency_1"] = adj
                connectivity["up_adjacency-1"] = adj

        elif t_normalized == "2-up-adjacency-0" or t == "2-up_adjacency-0":
            if adjacency_strategy == "pairs" and cycles is not None:
                ref_inc = get_inc(1)
                adj = build_node_cycle_adjacency_from_cycles(
                    cycles,
                    shape[0],
                    device=ref_inc.device,
                    dtype=ref_inc.dtype,
                    chunk_size=chunk_size,
                )
            else:
                inc1 = abs_sparse(get_inc(1))
                inc2 = abs_sparse(get_inc(2))
                B1B2 = safe_sparse_mm(inc1, inc2)
                adj = remove_diag_sparse(safe_sparse_mm(B1B2, B1B2.T))
            connectivity[t] = adj
            if legacy_keys:
                connectivity["2-up_adjacency-0"] = adj

        elif t_normalized == "up-laplacian-1" or t == "up_laplacian_1":
            inc2 = get_inc(2)

            if (
                adjacency_strategy == "pairs"
                and cycle_edges is not None
                and not signed
            ):
                uplap = build_edge_cycle_laplacian_from_cycle_edges(
                    cycle_edges,
                    shape[1],
                    device=inc2.device,
                    dtype=inc2.dtype,
                    chunk_size=chunk_size,
                )
            else:
                uplap = safe_sparse_mm(inc2, inc2.T)
                if not signed:
                    uplap = abs_sparse(uplap)

            if normalize_laplacians:
                uplap = gcn_normalize_sparse_tensor(uplap)

            connectivity[t] = uplap
            if legacy_keys:
                connectivity["up_laplacian_1"] = uplap
                connectivity["up_laplacian-1"] = uplap

        elif t_normalized == "down-laplacian-1" or t == "down_laplacian_1":
            inc1 = get_inc(1)
            downlap = safe_sparse_mm(inc1.T, inc1)
            if not signed:
                downlap = abs_sparse(downlap)

            if normalize_laplacians:
                downlap = gcn_normalize_sparse_tensor(downlap)

            connectivity[t] = downlap
            if legacy_keys:
                connectivity["down_laplacian_1"] = downlap
                connectivity["down_laplacian-1"] = downlap

        elif t_normalized == "up-adjacency-0" or t == "adjacency_0":
            inc1 = get_inc(1)
            adj = remove_diag_sparse(safe_sparse_mm(inc1, inc1.T))
            connectivity[t] = adj
            if legacy_keys:
                connectivity["adjacency_0"] = adj
                connectivity["up_adjacency-0"] = adj

        elif t_normalized == "down-adjacency-1" or t == "coadjacency_1":
            inc1 = get_inc(1)
            adj = remove_diag_sparse(safe_sparse_mm(inc1.T, inc1))
            connectivity[t] = adj
            if legacy_keys:
                connectivity["coadjacency_1"] = adj
                connectivity["down_adjacency-1"] = adj

        elif t_normalized == "down-adjacency-2" or t == "coadjacency_2":
            inc2 = abs_sparse(get_inc(2))
            adj = remove_diag_sparse(
                safe_sparse_mm(inc2.T, inc2), binarize=False
            )
            connectivity[t] = adj
            if legacy_keys:
                connectivity["coadjacency_2"] = adj
                connectivity["down_adjacency-2"] = adj

        elif t_normalized in {"up-incidence-0", "up-incidence-1"}:
            rank = int(t_normalized[-1]) + 1
            inc = get_inc(rank)
            inc = inc if signed else abs_sparse(inc)
            connectivity[t] = inc.T
            if legacy_keys:
                connectivity[f"up_incidence-{rank - 1}"] = inc.T

        elif t_normalized in {"down-incidence-1", "down-incidence-2"}:
            rank = int(t_normalized[-1])
            inc = get_inc(rank)
            inc = inc if signed else abs_sparse(inc)
            connectivity[t] = inc
            if legacy_keys:
                connectivity[f"down_incidence-{rank}"] = inc

        else:
            raise ValueError(f"Unsupported neighborhood: {t}")

    return connectivity


def get_combinatorial_complex_connectivity(
    complex, max_rank, neighborhoods=None
):
    r"""Get the connectivity matrices for the Combinatorial Complex.

    Parameters
    ----------
    complex : topnetx.CombinatorialComplex
        Cell complex.
    max_rank : int
        Maximum rank of the complex.
    neighborhoods : list, optional
        List of neighborhoods of interest.

    Returns
    -------
    dict
        Dictionary containing the connectivity matrices.
    """
    practical_shape = list(
        np.pad(list(complex.shape), (0, max_rank + 1 - len(complex.shape)))
    )
    connectivity = {}
    for rank_idx in range(max_rank + 1):
        for connectivity_info in [
            "incidence",
            "down_laplacian",
            "up_laplacian",
            "adjacency",
            "coadjacency",
            "hodge_laplacian",
        ]:
            try:
                if connectivity_info == "adjacency":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        from_sparse(
                            getattr(complex, f"{connectivity_info}_matrix")(
                                rank_idx, rank_idx + 1
                            )
                        )
                    )
                else:  # incidence
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        from_sparse(
                            getattr(complex, f"{connectivity_info}_matrix")(
                                rank_idx - 1, rank_idx
                            )
                        )
                    )
            except ValueError:
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
            except AttributeError:
                if connectivity_info == "incidence":
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx - 1],
                            n=practical_shape[rank_idx],
                        )
                    )
                else:
                    connectivity[f"{connectivity_info}_{rank_idx}"] = (
                        generate_zero_sparse_connectivity(
                            m=practical_shape[rank_idx],
                            n=practical_shape[rank_idx],
                        )
                    )
    if neighborhoods is not None:
        connectivity = select_neighborhoods_of_interest(
            connectivity, neighborhoods
        )
    connectivity["shape"] = practical_shape
    return connectivity


def select_neighborhoods_of_interest(connectivity, neighborhoods):
    """Select the neighborhoods of interest.

    Parameters
    ----------
    connectivity : dict
        Connectivity matrices generated by default.
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    dict
        Connectivity matrices of interest.
    """

    def generate_adjacency_from_laplacian(sparse_tensor):
        """Generate an adjacency matrix from a Laplacian matrix.

        Parameters
        ----------
        sparse_tensor : torch.sparse_coo_tensor
            Sparse tensor representing the Laplacian matrix.

        Returns
        -------
        torch.sparse_coo_tensor
            Sparse tensor representing the adjacency matrix.
        """
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()

        # Create a mask for non-diagonal elements
        mask = indices[0] != indices[1]

        # Filter indices and values based on the mask
        new_indices = indices[:, mask]
        new_values = values[mask]

        # Turn values to 1s
        new_values = new_values / new_values

        # Construct a new sparse tensor
        return torch.sparse_coo_tensor(
            new_indices, new_values, sparse_tensor.size()
        )

    useful_connectivity = {}
    for neighborhood in neighborhoods:
        src_rank = int(neighborhood.split("-")[-1])
        try:
            if (
                len(neighborhood.split("-")) == 2
                or neighborhood.split("-")[0] == "1"
            ):
                r = 1
                neighborhood_type = (
                    neighborhood.split("-")[0]
                    if neighborhood.split("-")[0] != "1"
                    else neighborhood.split("-")[1]
                )
                if "adjacency" in neighborhood_type:
                    useful_connectivity[neighborhood] = (
                        connectivity[f"adjacency_{src_rank}"]
                        if "up" in neighborhood_type
                        else connectivity[f"coadjacency_{src_rank}"]
                    )
                elif "laplacian" in neighborhood_type:
                    useful_connectivity[neighborhood] = connectivity[
                        f"{neighborhood_type}_{src_rank}"
                    ]

                elif "incidence" in neighborhood_type:
                    useful_connectivity[neighborhood] = (
                        connectivity[f"incidence_{src_rank + 1}"].T
                        if "up" in neighborhood_type
                        else connectivity[f"incidence_{src_rank}"]
                    )
            elif len(neighborhood.split("-")) == 3:
                r = int(neighborhood.split("-")[0])
                neighborhood_type = neighborhood.split("-")[1]
                if (
                    "adjacency" in neighborhood_type
                    or "laplacian" in neighborhood_type
                ):
                    direction, connectivity_type = neighborhood_type.split("_")
                    if direction == "up":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank + 1}"],
                            connectivity[f"incidence_{src_rank + 2}"],
                        )
                        for idx in range(src_rank + 3, src_rank + r + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Multiply the resulting matrix by its transpose to get the laplacian matrix
                        matrix = torch.sparse.mm(matrix, matrix.T)
                        # Turn all values to 1s
                        matrix = torch.sparse_coo_tensor(
                            matrix.indices(),
                            matrix.values() / matrix.values(),
                            matrix.size(),
                        )
                        # Generate the adjacency matrix from the laplacian if needed
                        useful_connectivity[neighborhood] = (
                            generate_adjacency_from_laplacian(matrix)
                            if "adjacency" in neighborhood_type
                            else matrix
                        )
                    elif direction == "down":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank - r + 1}"],
                            connectivity[f"incidence_{src_rank - r + 2}"],
                        )
                        for idx in range(src_rank - r + 3, src_rank + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Multiply the resulting matrix by its transpose to get the laplacian matrix
                        matrix = torch.sparse.mm(matrix.T, matrix)
                        # Turn all values to 1s
                        matrix = torch.sparse_coo_tensor(
                            matrix.indices(),
                            matrix.values() / matrix.values(),
                            matrix.size(),
                        )
                        # Generate the adjacency matrix from the laplacian if needed
                        useful_connectivity[neighborhood] = (
                            generate_adjacency_from_laplacian(matrix)
                            if "adjacency" in neighborhood_type
                            else matrix
                        )
                elif "incidence" in neighborhood_type:
                    direction, connectivity_type = neighborhood_type.split("_")
                    if direction == "up":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank + 1}"],
                            connectivity[f"incidence_{src_rank + 2}"],
                        )
                        for idx in range(src_rank + 3, src_rank + r + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Turn all values to 1s and transpose the matrix
                        useful_connectivity[neighborhood] = (
                            torch.sparse_coo_tensor(
                                matrix.indices(),
                                matrix.values() / matrix.values(),
                                matrix.size(),
                            ).T
                        )
                    elif direction == "down":
                        # Multiply consecutive incidence matrices up to getting the desired rank
                        matrix = torch.sparse.mm(
                            connectivity[f"incidence_{src_rank - r + 1}"],
                            connectivity[f"incidence_{src_rank - r + 2}"],
                        )
                        for idx in range(src_rank - r + 3, src_rank + 1):
                            matrix = torch.sparse.mm(
                                matrix, connectivity[f"incidence_{idx}"]
                            )
                        # Turn all values to 1s
                        useful_connectivity[neighborhood] = (
                            torch.sparse_coo_tensor(
                                matrix.indices(),
                                matrix.values() / matrix.values(),
                                matrix.size(),
                            )
                        )
                    elif direction == "virtualnode":
                        useful_connectivity[neighborhood] = connectivity[
                            f"incidence_{src_rank}"
                        ]

            else:
                useful_connectivity[neighborhood] = connectivity[neighborhood]
        except:  # noqa: E722
            raise ValueError(f"Invalid neighborhood {neighborhood}")  # noqa: B904
    for key in connectivity:
        if "incidence" in key and "-" not in key:
            useful_connectivity[key] = connectivity[key]
    for key, value in list(useful_connectivity.items()):
        if "incidence" in key and "-" not in key:
            continue
        try:
            neighborhood_type, rank, hop = _parse_neighborhood_target(key)
        except (IndexError, ValueError):
            continue
        for alias in _connectivity_aliases(neighborhood_type, rank, hop):
            useful_connectivity.setdefault(alias, value)
    return useful_connectivity


def generate_zero_sparse_connectivity(m, n):
    """Generate a zero sparse connectivity matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    torch.sparse_coo_tensor
        Zero sparse connectivity matrix.
    """
    return torch.sparse_coo_tensor((m, n)).coalesce()


def load_cell_complex_dataset(cfg):
    r"""Load cell complex datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.
    """
    raise NotImplementedError


def load_simplicial_dataset(cfg):
    """Load simplicial datasets.

    Parameters
    ----------
    cfg : DictConfig
        Configuration parameters.

    Returns
    -------
    torch_geometric.data.Data
        Simplicial dataset.
    """
    raise NotImplementedError


def load_manual_graph():
    """Create a manual graph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual graph.
    """
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the edges
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [5, 2],
        [5, 6],
        [6, 3],
        [5, 7],
        [2, 7],
        [0, 7],
    ]

    # Define the tetrahedrons
    tetrahedrons = [[0, 1, 2, 4]]

    # Add tetrahedrons
    for tetrahedron in tetrahedrons:
        for i in range(len(tetrahedron)):
            for j in range(i + 1, len(tetrahedron)):
                edges.append([tetrahedron[i], tetrahedron[j]])  # noqa: PERF401

    # Create a graph
    G = nx.Graph()

    # Add vertices
    G.add_nodes_from(vertices)

    # Add edges
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate feature from 0 to 9
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )


def load_manual_graph_second_structure():
    """Create a manual graph for testing purposes with updated edges and node features.

    Returns
    -------
    torch_geometric.data.Data
        A simple graph data object.
    """
    # Define the vertices (12 vertices, based on the highest index in edges)
    vertices = [i for i in range(12)]
    y = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

    # Updated edges
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (3, 4),
        (3, 6),
        (3, 9),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 0),
        (5, 7),
        (5, 10),
        (5, 11),
        (6, 9),
        (7, 8),
        (8, 6),
        (10, 11),
    ]

    # Create a graph
    G = nx.Graph()

    # Add vertices and edges
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    G.to_undirected()
    edge_list = torch.Tensor(list(G.edges())).T.long()

    # Generate updated features (example features for 12 nodes)
    x = (
        torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000, 200, 300, 400, 600])
        .unsqueeze(1)
        .float()
    )

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_list,
        num_nodes=len(vertices),
        y=torch.tensor(y),
    )
    return data


def ensure_serializable(obj):
    """Ensure that the object is serializable.

    Parameters
    ----------
    obj : object
        Object to ensure serializability.

    Returns
    -------
    object
        Object that is serializable.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = ensure_serializable(value)
        return obj
    elif isinstance(obj, list | tuple | omegaconf.listconfig.ListConfig):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return {ensure_serializable(item) for item in obj}
    elif isinstance(obj, str | int | float | bool | type(None)):
        return obj
    elif isinstance(obj, omegaconf.dictconfig.DictConfig):
        obj = omegaconf.OmegaConf.to_container(obj, resolve=False)
        return obj
    else:
        return None


def make_hash(o):
    """Make a hash from a dictionary, list, tuple or set to any level, that contains only other hashable types.

    Parameters
    ----------
    o : dict, list, tuple, set
        Object to hash.

    Returns
    -------
    int
        Hash of the object.
    """
    sha1 = hashlib.sha1()
    sha1.update(str.encode(str(o)))
    hash_as_hex = sha1.hexdigest()
    # Convert the hex back to int and restrict it to the relevant int range
    return int(hash_as_hex, 16) % 4294967295


def load_manual_hypergraph():
    """Create a manual hypergraph for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual hypergraph.
    """
    # Define the vertices (just 8 vertices)
    vertices = [i for i in range(8)]
    y = [0, 1, 1, 1, 0, 0, 0, 0]
    # Define the hyperedges
    hyperedges = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
        [3, 4],
        [4, 5],
        [4, 7],
        [5, 6],
        [6, 7],
    ]

    # Generate feature from 0 to 7
    x = torch.tensor([1, 5, 10, 50, 100, 500, 1000, 5000]).unsqueeze(1).float()
    labels = torch.tensor(y, dtype=torch.long)

    node_list = []
    edge_list = []

    for edge_idx, he in enumerate(hyperedges):
        cur_size = len(he)
        node_list += he
        edge_list += [edge_idx] * cur_size

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    incidence_hyperedges = torch.sparse_coo_tensor(
        edge_index,
        values=torch.ones(edge_index.shape[1]),
        size=(len(vertices), len(hyperedges)),
    )

    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        incidence_hyperedges=incidence_hyperedges,
    )


def load_manual_pointcloud(pos_to_x: bool = False):
    """Create a manual pointcloud for testing purposes.

    Parameters
    ----------
    pos_to_x : bool, optional
        If True, the positions are used as features.

    Returns
    -------
    torch_geometric.data.Data
        Manual pointcloud.
    """
    # Define the positions
    pos = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1],
            [20, 0, 0],
            [20, 0, 1],
            [20, 1, 0],
            [20, 1, 1],
            [30, 0, 0],
        ]
    ).float()

    if pos_to_x:
        return torch_geometric.data.Data(
            x=pos, pos=pos, num_nodes=pos.size(0), num_features=pos.size(1)
        )

    return torch_geometric.data.Data(
        pos=pos, num_nodes=pos.size(0), num_features=0
    )


def load_manual_points():
    """Create a manual point cloud for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual point cloud.
    """
    pos = torch.tensor(
        [
            [1.0, 1.0],
            [7.0, 0.0],
            [4.0, 6.0],
            [9.0, 6.0],
            [0.0, 14.0],
            [2.0, 19.0],
            [9.0, 17.0],
        ],
        dtype=torch.float,
    )
    y = torch.randint(0, 2, (pos.shape[0],), dtype=torch.float)
    return torch_geometric.data.Data(x=pos, y=y, complex_dim=0)


def load_manual_simplicial_complex():
    """Create a manual simplicial complex for testing purposes.

    Returns
    -------
    torch_geometric.data.Data
        Manual simplicial complex.
    """
    num_feats = 2
    one_cells = [i for i in range(5)]
    two_cells = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [0, 4], [2, 4]]
    three_cells = [[0, 1, 2], [1, 2, 3], [0, 2, 4]]
    incidence_1 = [
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
    ]
    incidence_2 = [
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]

    y = [1]

    return torch_geometric.data.Data(
        x_0=torch.rand(len(one_cells), num_feats),
        x_1=torch.rand(len(two_cells), num_feats),
        x_2=torch.rand(len(three_cells), num_feats),
        incidence_0=torch.zeros((1, 5)).to_sparse(),
        adjacency_1=torch.zeros((len(one_cells), len(one_cells))).to_sparse(),
        adjacency_2=torch.zeros((len(two_cells), len(two_cells))).to_sparse(),
        adjacency_0=torch.zeros((5, 5)).to_sparse(),
        incidence_1=torch.tensor(incidence_1).to_sparse(),
        incidence_2=torch.tensor(incidence_2).to_sparse(),
        num_nodes=len(one_cells),
        y=torch.tensor(y),
    )


def data2simplicial(data):
    """
    Convert a data dictionary into a SimplicialComplex object.

    Parameters
    ----------
    data : dict
        A dictionary containing at least 'incidence_0', 'adjacency_0', 'incidence_1',
        'incidence_2', and optionally 'incidence_3' tensors.

    Returns
    -------
    SimplicialComplex
        A SimplicialComplex object constructed from nodes, edges, triangles, and tetrahedrons.
    """
    sc = SimplicialComplex()

    # Nodes as single-element lists
    nodes = [[i] for i in range(data["incidence_0"].shape[1])]

    # Convert edges to a list of pairs
    edges = torch_geometric.utils.remove_self_loops(
        data["adjacency_0"].indices()
    )[0].T.tolist()

    # Detect triangles if incidence_1 and incidence_2 exist
    triangles = (
        find_triangles(data["incidence_1"], data["incidence_2"])
        if "incidence_1" in data and "incidence_2" in data
        else []
    )

    # Detect tetrahedrons if incidence_3 exists
    tetrahedrons = (
        find_tetrahedrons(
            data["incidence_1"], data["incidence_2"], data["incidence_3"]
        )
        if "incidence_3" in data
        else []
    )

    # Add simplices to the complex
    sc.add_simplices_from(nodes)
    sc.add_simplices_from(edges)
    sc.add_simplices_from(triangles)
    sc.add_simplices_from(tetrahedrons)

    return sc


def find_triangles(incidence_1, incidence_2):
    """
    Identify triangles in the simplicial complex based on incidence matrices.

    Parameters
    ----------
    incidence_1 : torch.Tensor
        Incidence matrix of edges.
    incidence_2 : torch.Tensor
        Incidence matrix of triangles.

    Returns
    -------
    list of list
        List of triangles, where each triangle is a list of three node indices.
    """
    triangles = (incidence_1 @ incidence_2).indices()
    unique_triangles = torch.unique(triangles[1])
    triangle_list = [
        [j.item() for j in triangles[0][torch.where(triangles[1] == i)[0]]]
        for i in unique_triangles
    ]
    return triangle_list


def find_tetrahedrons(incidence_1, incidence_2, incidence_3):
    """
    Identify tetrahedrons in the simplicial complex.

    Parameters
    ----------
    incidence_1 : torch.Tensor
        Incidence matrix of edges.
    incidence_2 : torch.Tensor
        Incidence matrix of triangles.
    incidence_3 : torch.Tensor
        Incidence matrix of tetrahedrons.

    Returns
    -------
    list of list
        List of tetrahedrons, where each is represented as a list of four node indices.
    """
    tetrahedrons = (incidence_1 @ incidence_2 @ incidence_3).indices()
    unique_tetrahedrons = torch.unique(tetrahedrons[1])
    tetrahedron_list = [
        [
            j.item()
            for j in tetrahedrons[0][torch.where(tetrahedrons[1] == i)[0]]
        ]
        for i in unique_tetrahedrons
    ]
    return tetrahedron_list
