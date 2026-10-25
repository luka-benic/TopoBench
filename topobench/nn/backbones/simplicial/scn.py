"""Matrix-free Simplex Convolutional Network."""

from collections.abc import Callable

import torch
from topomodelx.base.conv import Conv

from topobench.nn.backbones.simplicial.incidence_operators import (
    BoundaryOperator,
    UnsignedHodgeOperator,
)


class SCN2MatrixFreeLayer(torch.nn.Module):
    """SCN layer operating directly on boundary incidences.

    Parameters
    ----------
    in_channels_0 : int
        Number of node channels.
    in_channels_1 : int
        Number of edge channels.
    in_channels_2 : int
        Number of face channels.
    """

    def __init__(
        self,
        in_channels_0: int,
        in_channels_1: int,
        in_channels_2: int,
    ) -> None:
        super().__init__()
        self.conv_0_to_0 = Conv(
            in_channels=in_channels_0, out_channels=in_channels_0
        )
        self.conv_1_to_1 = Conv(
            in_channels=in_channels_1, out_channels=in_channels_1
        )
        self.conv_2_to_2 = Conv(
            in_channels=in_channels_2, out_channels=in_channels_2
        )

    @staticmethod
    def _convolve(
        x: torch.Tensor,
        conv: Conv,
        operator: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply a learned transform followed by a Hodge operator.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        conv : Conv
            Learned channel transformation.
        operator : callable
            Matrix-free Hodge operator.

        Returns
        -------
        torch.Tensor
            Convolved features.
        """
        return operator(torch.mm(x, conv.weight))

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        operators: tuple[UnsignedHodgeOperator, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one matrix-free SCN layer.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features.
        x_1 : torch.Tensor
            Edge features.
        x_2 : torch.Tensor
            Face features.
        operators : tuple of UnsignedHodgeOperator
            Operators for ranks zero, one, and two.

        Returns
        -------
        tuple of torch.Tensor
            Updated node, edge, and face features.
        """
        hodge_0, hodge_1, hodge_2 = operators
        x_0 = torch.nn.functional.relu(
            self._convolve(x_0, self.conv_0_to_0, hodge_0)
        )
        x_1 = torch.nn.functional.relu(
            self._convolve(x_1, self.conv_1_to_1, hodge_1)
        )
        x_2 = torch.nn.functional.relu(
            self._convolve(x_2, self.conv_2_to_2, hodge_2)
        )
        return x_0, x_1, x_2


class SCN2MatrixFree(torch.nn.Module):
    """SCN2 implementation that never constructs Hodge Laplacians.

    Parameters
    ----------
    in_channels_0 : int
        Number of node channels.
    in_channels_1 : int
        Number of edge channels.
    in_channels_2 : int
        Number of face channels.
    n_layers : int, optional
        Number of message-passing layers.
    """

    def __init__(
        self,
        in_channels_0: int,
        in_channels_1: int,
        in_channels_2: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            SCN2MatrixFreeLayer(
                in_channels_0=in_channels_0,
                in_channels_1=in_channels_1,
                in_channels_2=in_channels_2,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        incidence_1: torch.Tensor,
        incidence_2: torch.Tensor,
        incidence_3: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply SCN layers using normalized incidence-based operators.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features.
        x_1 : torch.Tensor
            Edge features.
        x_2 : torch.Tensor
            Face features.
        incidence_1 : torch.Tensor
            Node-to-edge boundary matrix.
        incidence_2 : torch.Tensor
            Edge-to-face boundary matrix.
        incidence_3 : torch.Tensor, optional
            Face-to-tetrahedron boundary matrix when rank-3 cells exist.

        Returns
        -------
        tuple of torch.Tensor
            Updated node, edge, and face features.
        """
        boundary_1 = BoundaryOperator(incidence_1)
        boundary_2 = BoundaryOperator(incidence_2)
        boundary_3 = (
            BoundaryOperator(incidence_3) if incidence_3 is not None else None
        )
        operators = (
            UnsignedHodgeOperator(
                lower=None,
                upper=boundary_1,
                num_simplices=x_0.size(0),
                dtype=x_0.dtype,
                device=x_0.device,
                normalize=True,
            ),
            UnsignedHodgeOperator(
                lower=boundary_1,
                upper=boundary_2,
                num_simplices=x_1.size(0),
                dtype=x_1.dtype,
                device=x_1.device,
                normalize=True,
            ),
            UnsignedHodgeOperator(
                lower=boundary_2,
                upper=boundary_3,
                num_simplices=x_2.size(0),
                dtype=x_2.dtype,
                device=x_2.device,
                normalize=True,
            ),
        )

        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, operators)
        return x_0, x_1, x_2
