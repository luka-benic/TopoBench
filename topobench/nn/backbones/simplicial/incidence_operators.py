"""Matrix-free simplicial operators built from boundary incidences."""

from __future__ import annotations

from collections.abc import Callable

import torch


class BoundaryOperator:
    r"""Apply lower and upper Laplacians without materializing them.

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse boundary matrix with lower-rank simplices in rows and
        higher-rank simplices in columns.
    """

    def __init__(self, incidence: torch.Tensor) -> None:
        if incidence.layout != torch.sparse_coo:
            incidence = incidence.to_sparse_coo()
        self.incidence = incidence.coalesce()
        self.transpose = self.incidence.transpose(0, 1).coalesce()

    def up(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`BB^\top` to a lower-rank signal.

        Parameters
        ----------
        x : torch.Tensor
            Lower-rank signal.

        Returns
        -------
        torch.Tensor
            Transformed lower-rank signal.
        """
        return torch.sparse.mm(
            self.incidence,
            torch.sparse.mm(self.transpose, x),
        )

    def down(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`B^\top B` to a higher-rank signal.

        Parameters
        ----------
        x : torch.Tensor
            Higher-rank signal.

        Returns
        -------
        torch.Tensor
            Transformed higher-rank signal.
        """
        return torch.sparse.mm(
            self.transpose,
            torch.sparse.mm(self.incidence, x),
        )

    def up_diagonal(self) -> torch.Tensor:
        r"""Return the diagonal of :math:`BB^\top`.

        Returns
        -------
        torch.Tensor
            Operator diagonal.
        """
        indices = self.incidence.indices()[0]
        values = self.incidence.values()
        diagonal = torch.zeros(
            self.incidence.size(0),
            dtype=values.dtype,
            device=values.device,
        )
        diagonal.scatter_add_(0, indices, values.square())
        return diagonal


class UnsignedHodgeOperator:
    r"""Apply TopoBench's unsigned Hodge convention matrix-free.

    The explicit selective lifting computes the signed Hodge Laplacian and
    then takes its elementwise absolute value. For an abstract simplicial
    complex, upper-adjacent simplex pairs are also lower adjacent and their
    signed terms cancel. This class reproduces that result from unsigned
    boundary incidences.

    Parameters
    ----------
    lower : BoundaryOperator or None
        Boundary operator whose columns are the target-rank simplices.
    upper : BoundaryOperator or None
        Boundary operator whose rows are the target-rank simplices.
    num_simplices : int
        Number of target-rank simplices.
    dtype : torch.dtype
        Signal dtype.
    device : torch.device
        Signal device.
    normalize : bool, optional
        Apply symmetric row-sum normalization.
    """

    def __init__(
        self,
        lower: BoundaryOperator | None,
        upper: BoundaryOperator | None,
        num_simplices: int,
        dtype: torch.dtype,
        device: torch.device,
        normalize: bool = False,
    ) -> None:
        if lower is None and upper is None:
            raise ValueError("At least one boundary operator is required.")
        self.lower = lower
        self.upper = upper
        self.normalizer = None
        if normalize:
            ones = torch.ones((num_simplices, 1), dtype=dtype, device=device)
            degree = self._apply_raw(ones).squeeze(1)
            normalizer = torch.zeros_like(degree)
            nonzero = degree > 0
            normalizer[nonzero] = torch.rsqrt(degree[nonzero])
            self.normalizer = normalizer

    def _apply_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the unnormalized operator.

        Parameters
        ----------
        x : torch.Tensor
            Target-rank signal.

        Returns
        -------
        torch.Tensor
            Unnormalized transformed signal.
        """
        if self.lower is None:
            return self.upper.up(x)
        if self.upper is None:
            return self.lower.down(x)

        upper = self.upper.up(x)
        upper_diagonal = self.upper.up_diagonal().unsqueeze(1)
        return self.lower.down(x) - upper + 2 * upper_diagonal * x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the optionally normalized unsigned Hodge operator.

        Parameters
        ----------
        x : torch.Tensor
            Target-rank signal.

        Returns
        -------
        torch.Tensor
            Transformed target-rank signal.
        """
        if self.normalizer is None:
            return self._apply_raw(x)
        scaled = self.normalizer.unsqueeze(1) * x
        return self.normalizer.unsqueeze(1) * self._apply_raw(scaled)


def zero_operator(x: torch.Tensor) -> torch.Tensor:
    """Return a zero signal with the same shape as the input.

    Parameters
    ----------
    x : torch.Tensor
        Input signal.

    Returns
    -------
    torch.Tensor
        Zero signal.
    """
    return torch.zeros_like(x)


def operator_powers(
    operator: Callable[[torch.Tensor], torch.Tensor],
    order: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """Stack consecutive matrix-free operator powers applied to a signal.

    Parameters
    ----------
    operator : callable
        Matrix-free linear operator.
    order : int
        Number of consecutive powers.
    x : torch.Tensor
        Input signal.

    Returns
    -------
    torch.Tensor
        Operator outputs stacked along the last dimension.
    """
    outputs = []
    current = x
    for _ in range(order):
        current = operator(current)
        outputs.append(current)
    return torch.stack(outputs, dim=2)
