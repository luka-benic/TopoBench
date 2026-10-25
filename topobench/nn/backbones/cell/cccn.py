"""Convolutional Cell Convolutional Network (CCCN) model."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CCCN(nn.Module):
    r"""CCCN model.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    n_layers : int, optional
        Number of layers (default: 2).
    dropout : float, optional
        Dropout rate (default: 0).
    last_act : bool, optional
        If True, the last activation function is applied (default: False).
    use_pre_normalized_operators : bool, optional
        If True, adjacency operators are assumed to be pre-normalized.
    """

    def __init__(
        self,
        in_channels,
        n_layers=2,
        dropout=0.0,
        last_act=False,
        use_pre_normalized_operators=False,
    ):
        super().__init__()
        self.d = dropout
        self.convs = nn.ModuleList()
        self.last_act = last_act
        self.use_pre_normalized_operators = use_pre_normalized_operators

        normalize = not self.use_pre_normalized_operators
        for _ in range(n_layers):
            self.convs.append(
                CW(in_channels, in_channels, normalize=normalize)
            )

    def forward(self, x, Ld, Lu):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        Ld : torch.Tensor
            Domain adjacency matrix.
        Lu : torch.Tensor
            Label adjacency matrix.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for i, c in enumerate(self.convs):
            x = c(F.dropout(x, p=self.d, training=self.training), Lu, Ld)
            if i == len(self.convs) and self.last_act is False:
                break
            x = x.relu()
        return x


class CW(nn.Module):
    r"""Layer of the CCCN model.

    Parameters
    ----------
    F_in : int
        Number of input channels.
    F_out : int
        Number of output channels.
    normalize : bool, optional
        Whether graph convolution layers should normalize adjacency values.
    """

    def __init__(self, F_in, F_out, normalize=True):
        super().__init__()
        self.har = nn.Linear(F_in, F_out)
        self.sol = GCNConv(
            F_in, F_out, add_self_loops=False, normalize=normalize
        )
        self.irr = GCNConv(
            F_in, F_out, add_self_loops=False, normalize=normalize
        )

    def forward(self, xe, Lu, Ld):
        r"""Forward pass.

        Parameters
        ----------
        xe : torch.Tensor
            Input tensor.
        Lu : torch.Tensor
            Domain adjacency matrix.
        Ld : torch.Tensor
            Label adjacency matrix.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        z_h = self.har(xe)
        z_s = self.sol(xe, Lu)
        z_i = self.irr(xe, Ld)
        return z_h + z_s + z_i
