"""Campo gravitacional que deforma a metrica baseado na massa dos tokens."""

import torch
import torch.nn as nn


class GravityField(nn.Module):
    """Modifica G(x) baseado na massa dos tokens.

    Tokens com alta informacao adicionam curvatura local via
    kernel Gaussiano ponderado pela massa. Usa aproximacao RFF
    (Random Fourier Features) para O(T*R) em vez de O(T^2).

    Args:
        d_manifold: Dimensao do manifold.
        strength: Forca da gravidade.
        sigma: Largura do kernel Gaussiano.
        n_rff: Numero de random Fourier features (64-128).
    """

    def __init__(
        self,
        d_manifold: int,
        strength: float = 0.1,
        sigma: float = 0.5,
        n_rff: int = 64,
    ):
        super().__init__()
        self.d_manifold = d_manifold
        self.strength = strength
        self.sigma = sigma
        self.n_rff = n_rff

        self.mass_net = nn.Sequential(
            nn.Linear(d_manifold, d_manifold),
            nn.ReLU(),
            nn.Linear(d_manifold, 1),
            nn.Softplus(),
        )

        # W ~ N(0, 1/sigma^2), fixo (nao aprendivel)
        W = torch.randn(d_manifold, n_rff) / sigma
        b = torch.rand(n_rff) * 2 * torch.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def _rff_features(self, coords: torch.Tensor) -> torch.Tensor:
        """Computa features RFF para coords.

        phi(x) = sqrt(2/R) * cos(x @ W + b)
        Aproxima o kernel Gaussiano: phi(x)^T phi(y) ~ exp(-||x-y||^2 / 2sigma^2)

        Args:
            coords: [..., d_manifold]

        Returns:
            Tensor [..., n_rff]
        """
        proj = coords @ self.W + self.b
        return (2.0 / self.n_rff) ** 0.5 * torch.cos(proj)

    def compute_mass(self, coords: torch.Tensor) -> torch.Tensor:
        """Computa massa de cada token no manifold.

        Args:
            coords: [B, T, d_manifold]

        Returns:
            Tensor [B, T, 1] com massa >= 0 por token.
        """
        return self.mass_net(coords)

    def deform_metric(
        self,
        G: torch.Tensor,
        coords: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Deforma G(x) pela gravidade dos tokens via RFF.

        Cada token com massa > 0 adiciona curvatura local.
        Equivalente a G_grav(x) = G(x) + strength * sum_j mass_j * K(x, x_j) * I
        onde K e o kernel Gaussiano aproximado por RFF.

        Complexidade: O(T * R) em vez de O(T^2 * D).

        Args:
            G: [B, T, D, D] metrica base.
            coords: [B, T, D] coordenadas dos tokens.
            mass: [B, T, 1] massa de cada token.

        Returns:
            Tensor [B, T, D, D] com metrica deformada.
        """
        B, T, D = coords.shape

        phi = self._rff_features(coords)  # [B, T, R]

        # Ponderar features pela massa de cada token
        phi_weighted = phi * mass  # [B, T, R]

        # Influencia gravitacional acumulada via produto interno
        # grav[b, t] = phi[b,t] . sum_s phi_weighted[b,s]
        phi_sum = phi_weighted.sum(dim=1, keepdim=True)  # [B, 1, R]
        grav_influence = (phi * phi_sum).sum(dim=-1, keepdim=True)  # [B, T, 1]
        grav_influence = grav_influence.unsqueeze(-1)  # [B, T, 1, 1]

        I = torch.eye(D, device=G.device, dtype=G.dtype)
        return G + self.strength * grav_influence * I
