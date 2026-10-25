"""Unit tests for simplicial model wrappers."""

import pytest
from topomodelx.nn.simplicial.san import SAN
from topomodelx.nn.simplicial.sccn import SCCN
from topomodelx.nn.simplicial.sccnn import SCCNN

from topobench.nn.backbones.simplicial import SCCNNCustom, SCN2MatrixFree
from topobench.nn.wrappers import (
    SANWrapper,
    SCCNNCustomWrapper,
    SCCNNWrapper,
    SCCNWrapper,
    SCNWrapper,
)


class TestSimplicialWrappers:
    r"""Test simplicial model wrappers.

    Test all simplicial wrappers.
    """

    @pytest.mark.parametrize(
        "backbone, wrapper_class",
        [(SCCNN, SCCNNWrapper), (SCCNNCustom, SCCNNCustomWrapper)],
    )
    def test_SCCNNWrapper(self, sg1_clique_lifted, backbone, wrapper_class):
        """Test SCCNNWrapper.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with SimlicialCliqueLifting.
        """
        data = sg1_clique_lifted
        out_dim = 4
        conv_order = 1
        sc_order = 3
        init_args = (
            (data.x_0.shape[1], data.x_1.shape[1], data.x_2.shape[1]),
            (out_dim, out_dim, out_dim),
            conv_order,
            sc_order,
        )

        wrapper = wrapper_class(
            backbone(*init_args),
            out_channels=out_dim,
            num_cell_dimensions=3,
        )
        out = wrapper(data)
        # Assert keys in output
        for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
            assert key in out

    def test_SANWrapper(self, sg1_clique_lifted):
        """Test SANWrapper.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with SimlicialCliqueLifting.
        """
        data = sg1_clique_lifted
        out_dim = data.x_0.shape[1]
        hidden_channels = data.x_0.shape[1]

        wrapper = SANWrapper(
            SAN(data.x_0.shape[1], hidden_channels),
            out_channels=out_dim,
            num_cell_dimensions=3,
        )
        out = wrapper(data)
        # Assert keys in output
        for key in ["labels", "batch_0", "x_0", "x_1"]:
            assert key in out

    def test_SCNWrapper(self, sg1_clique_lifted):
        """Test SCNWrapper.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with SimlicialCliqueLifting.
        """
        data = sg1_clique_lifted
        out_dim = data.x_0.shape[1]

        wrapper = SCNWrapper(
            SCN2MatrixFree(
                data.x_0.shape[1],
                data.x_1.shape[1],
                data.x_2.shape[1],
            ),
            out_channels=out_dim,
            num_cell_dimensions=3,
        )
        out = wrapper(data)
        # Assert keys in output
        for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
            assert key in out

    def test_SCCNWrapper(self, sg1_clique_lifted):
        """Test SCCNWrapper.

        Parameters
        ----------
        sg1_clique_lifted : torch_geometric.data.Data
            A fixture of simple graph 1 lifted with SimlicialCliqueLifting.
        """
        data = sg1_clique_lifted
        out_dim = data.x_0.shape[1]
        max_rank = 2

        wrapper = SCCNWrapper(
            SCCN(data.x_0.shape[1], max_rank),
            out_channels=out_dim,
            num_cell_dimensions=3,
        )
        out = wrapper(data)
        # Assert keys in output
        for key in ["labels", "batch_0", "x_0", "x_1", "x_2"]:
            assert key in out
