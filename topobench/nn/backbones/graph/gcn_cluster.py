"""Cluster-GCN model wrapper backbone."""

from typing import Final

from torch_geometric.nn.conv import ClusterGCNConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN


class ClusterGCN(BasicGNN):
    r"""Cluster-GCN model wrapper using :class:`~torch_geometric.nn.conv.ClusterGCNConv`."""

    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        """Initialize a ClusterGCNConv layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        MessagePassing
            The initialized ClusterGCNConv layer.
        """
        return ClusterGCNConv(in_channels, out_channels, **kwargs)
