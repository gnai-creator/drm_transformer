"""Tensor metrico SPD aprendido G(x) via decomposicao de Cholesky."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricNet(nn.Module):
    """Computa tensor metrico SPD dependente de posicao G(x).

    Garante G(x) SPD via Cholesky: G = L @ L^T onde L e triangular
    inferior com diagonal positiva.

    Args:
        dim: Dimensao do manifold.
        hidden: Dimensao da camada oculta do MLP.
    """

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.n_chol = dim * (dim + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_chol),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        rows, cols = torch.tril_indices(dim, dim)
        self.register_buffer("tril_row", rows)
        self.register_buffer("tril_col", cols)
        diag_idx = torch.arange(dim)
        self.register_buffer("diag_idx", diag_idx)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Computa G(x) para cada ponto.

        Args:
            coords: [..., dim] coordenadas no manifold.

        Returns:
            Tensor [..., dim, dim] metrico SPD por ponto.
        """
        raw = self.net(coords)

        if torch.isnan(raw).any():
            batch_shape = coords.shape[:-1]
            return torch.eye(
                self.dim, device=coords.device, dtype=coords.dtype,
            ).expand(*batch_shape, -1, -1).clone()

        batch_shape = raw.shape[:-1]
        L = torch.zeros(
            *batch_shape, self.dim, self.dim,
            device=raw.device, dtype=raw.dtype,
        )
        L[..., self.tril_row, self.tril_col] = raw

        diag_vals = F.softplus(L[..., self.diag_idx, self.diag_idx]) + 1e-3
        L[..., self.diag_idx, self.diag_idx] = diag_vals.to(L.dtype)

        G = torch.matmul(L, L.transpose(-1, -2))

        return G
