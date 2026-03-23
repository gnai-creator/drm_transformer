"""Campo gravitacional que deforma a metrica baseado na massa dos tokens."""

import torch
import torch.nn as nn


class GravityField(nn.Module):
    """Modifica G(x) baseado na massa dos tokens.

    Tokens com alta informacao adicionam curvatura local via
    kernel Gaussiano ponderado pela massa.

    Args:
        d_manifold: Dimensao do manifold.
        strength: Forca da gravidade.
    """

    def __init__(self, d_manifold: int, strength: float = 0.1):
        super().__init__()
        self.d_manifold = d_manifold
        self.strength = strength

        self.mass_net = nn.Sequential(
            nn.Linear(d_manifold, d_manifold),
            nn.ReLU(),
            nn.Linear(d_manifold, 1),
            nn.Softplus(),
        )

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
        sigma: float = 0.5,
    ) -> torch.Tensor:
        """Deforma G(x) pela gravidade dos tokens.

        Cada token com massa > 0 adiciona curvatura local via
        kernel Gaussiano: G_grav(x) = G(x) + strength * sum_j mass_j * K(x, x_j) * I.

        Args:
            G: [B, T, D, D] metrica base.
            coords: [B, T, D] coordenadas dos tokens.
            mass: [B, T, 1] massa de cada token.
            sigma: Largura do kernel gravitacional.

        Returns:
            Tensor [B, T, D, D] com metrica deformada.
        """
        B, T, D = coords.shape

        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        kernel = torch.exp(-dist_sq / (2 * sigma ** 2))

        grav_influence = (kernel * mass.squeeze(-1).unsqueeze(1)).sum(dim=-1)
        grav_influence = grav_influence.unsqueeze(-1).unsqueeze(-1)

        I = torch.eye(D, device=G.device, dtype=G.dtype)
        G_grav = G + self.strength * grav_influence * I

        return G_grav
