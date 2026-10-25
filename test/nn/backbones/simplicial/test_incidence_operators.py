"""Tests for matrix-free simplicial operators."""

import networkx as nx
import pytest
import torch
from topomodelx.nn.simplicial.scn2 import SCN2

from topobench.data.utils.utils import (
    get_simplicial_connectivity_from_incidences_selective,
)
from topobench.nn.backbones.simplicial import SCN2MatrixFree
from topobench.nn.backbones.simplicial.incidence_operators import (
    BoundaryOperator,
    UnsignedHodgeOperator,
    zero_operator,
)
from topobench.nn.backbones.simplicial.sccnn import SCCNNLayer
from topobench.transforms.liftings.graph2simplicial._clique_utils import (
    build_clique_complex_incidences,
)


@pytest.fixture(params=[2, 3])
def unsigned_connectivity(request):
    """Build an unsigned rank-two complex with overlapping triangles."""
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ]
    )
    incidences, _, shape, _ = build_clique_complex_incidences(
        graph=graph,
        complex_dim=request.param,
        signed=True,
        num_nodes=5,
    )
    neighborhoods = [
        "hodge_laplacian-0",
        "hodge_laplacian-1",
        "hodge_laplacian-2",
        "down_laplacian-1",
        "up_laplacian-1",
        "down_laplacian-2",
        "up_laplacian-2",
    ]
    return get_simplicial_connectivity_from_incidences_selective(
        incidences=incidences,
        shape=shape,
        max_rank=request.param,
        neighborhoods=neighborhoods,
        signed=False,
    )


def _normalized(matrix):
    dense = matrix.to_dense()
    degree = dense.abs().sum(dim=1)
    normalizer = torch.zeros_like(degree)
    nonzero = degree > 0
    normalizer[nonzero] = torch.rsqrt(degree[nonzero])
    return (normalizer[:, None] * dense * normalizer[None, :]).to_sparse()


def test_boundary_operators_match_explicit_laplacians(
    unsigned_connectivity,
):
    """Compare incidence products with every explicit SCCNN operator."""
    connectivity = unsigned_connectivity
    boundary_1 = BoundaryOperator(connectivity["incidence_1"])
    boundary_2 = BoundaryOperator(connectivity["incidence_2"])
    x_0 = torch.randn(connectivity["shape"][0], 4)
    x_1 = torch.randn(connectivity["shape"][1], 4)
    x_2 = torch.randn(connectivity["shape"][2], 4)

    assert torch.allclose(
        boundary_1.up(x_0),
        torch.sparse.mm(connectivity["hodge_laplacian_0"], x_0),
        atol=1e-6,
    )
    assert torch.allclose(
        boundary_1.down(x_1),
        torch.sparse.mm(connectivity["down_laplacian_1"], x_1),
        atol=1e-6,
    )
    assert torch.allclose(
        boundary_2.up(x_1),
        torch.sparse.mm(connectivity["up_laplacian_1"], x_1),
        atol=1e-6,
    )
    assert torch.allclose(
        boundary_2.down(x_2),
        torch.sparse.mm(connectivity["down_laplacian_2"], x_2),
        atol=1e-6,
    )


def test_unsigned_hodge_operators_match_explicit_normalized_actions(
    unsigned_connectivity,
):
    """Verify exact SCN behavior without explicit Hodge matrices."""
    connectivity = unsigned_connectivity
    boundary_1 = BoundaryOperator(connectivity["incidence_1"])
    boundary_2 = BoundaryOperator(connectivity["incidence_2"])
    boundary_3 = (
        BoundaryOperator(connectivity["incidence_3"])
        if "incidence_3" in connectivity
        else None
    )
    boundaries = (
        (None, boundary_1),
        (boundary_1, boundary_2),
        (boundary_2, boundary_3),
    )

    for rank, (lower, upper) in enumerate(boundaries):
        x = torch.randn(connectivity["shape"][rank], 4)
        operator = UnsignedHodgeOperator(
            lower=lower,
            upper=upper,
            num_simplices=x.size(0),
            dtype=x.dtype,
            device=x.device,
            normalize=True,
        )
        explicit = _normalized(connectivity[f"hodge_laplacian_{rank}"])
        assert torch.allclose(
            operator(x), torch.sparse.mm(explicit, x), atol=1e-6
        )


def test_matrix_free_scn_matches_explicit_scn(unsigned_connectivity):
    """Compare full SCN outputs after copying identical parameters."""
    connectivity = unsigned_connectivity
    channels = 4
    explicit = SCN2(channels, channels, channels, n_layers=1)
    matrix_free = SCN2MatrixFree(channels, channels, channels, n_layers=1)
    matrix_free.load_state_dict(explicit.state_dict())
    x = [
        torch.randn(connectivity["shape"][rank], channels, requires_grad=True)
        for rank in range(3)
    ]
    laplacians = [
        _normalized(connectivity[f"hodge_laplacian_{rank}"])
        for rank in range(3)
    ]

    expected = explicit(*x, *laplacians)
    matrix_free_x = [value.detach().clone().requires_grad_() for value in x]
    actual = matrix_free(
        *matrix_free_x,
        connectivity["incidence_1"],
        connectivity["incidence_2"],
        connectivity.get("incidence_3"),
    )

    for expected_rank, actual_rank in zip(expected, actual, strict=True):
        assert torch.allclose(expected_rank, actual_rank, atol=1e-6)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    for expected_x, actual_x in zip(x, matrix_free_x, strict=True):
        torch.testing.assert_close(
            expected_x.grad, actual_x.grad, atol=3e-6, rtol=1e-5
        )
    for expected_param, actual_param in zip(
        explicit.parameters(), matrix_free.parameters(), strict=True
    ):
        torch.testing.assert_close(
            expected_param.grad, actual_param.grad, atol=3e-6, rtol=1e-5
        )


def test_matrix_free_sccnn_layer_matches_explicit_laplacians(
    unsigned_connectivity,
):
    """Compare a complete SCCNN layer with explicit sparse products."""
    connectivity = unsigned_connectivity
    channels = 4
    layer = SCCNNLayer(
        in_channels=(channels, channels, channels),
        out_channels=(channels, channels, channels),
        conv_order=2,
        sc_order=3,
        update_func="sigmoid",
    )
    x = tuple(
        torch.randn(connectivity["shape"][rank], channels) for rank in range(3)
    )
    incidence = (
        connectivity["incidence_1"],
        connectivity["incidence_2"],
    )

    explicit_matrices = (
        connectivity["hodge_laplacian_0"],
        connectivity["down_laplacian_1"],
        connectivity["up_laplacian_1"],
        connectivity["down_laplacian_2"],
        connectivity["up_laplacian_2"],
    )
    explicit_operators = tuple(
        lambda current, matrix=matrix: torch.sparse.mm(matrix, current)
        for matrix in explicit_matrices
    )

    boundary_1 = BoundaryOperator(incidence[0])
    boundary_2 = BoundaryOperator(incidence[1])
    boundary_3 = (
        BoundaryOperator(connectivity["incidence_3"])
        if "incidence_3" in connectivity
        else None
    )
    matrix_free_operators = (
        boundary_1.up,
        boundary_1.down,
        boundary_2.up,
        boundary_2.down,
        boundary_3.up if boundary_3 is not None else zero_operator,
    )

    expected = layer(x, explicit_operators, incidence)
    actual = layer(x, matrix_free_operators, incidence)

    for expected_rank, actual_rank in zip(expected, actual, strict=True):
        assert torch.allclose(expected_rank, actual_rank, atol=1e-6)
