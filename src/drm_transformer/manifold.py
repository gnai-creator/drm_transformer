"""Projecao entre espaco de embeddings e manifold DRM, e gamma-scaling."""

import torch
import torch.nn as nn


class ManifoldProjection(nn.Module):
    """Projeta embeddings do espaco d_model para o manifold d_manifold.

    Usa Sigmoid para manter coordenadas em [0, 1]^d_manifold,
    garantindo manifold compacto para convergencia toroidal.

    Args:
        d_model: Dimensao do embedding.
        d_manifold: Dimensao do manifold.
    """

    def __init__(self, d_model: int, d_manifold: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_manifold * 4),
            nn.GELU(),
            nn.Linear(d_manifold * 4, d_manifold),
        )
        self.inv_proj = nn.Linear(d_manifold, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projeta embeddings para o manifold.

        Args:
            x: [B, T, d_model] embeddings.

        Returns:
            Coordenadas [B, T, d_manifold] em [0, 1].
        """
        return torch.sigmoid(self.proj(x))

    def inverse(self, coords: torch.Tensor) -> torch.Tensor:
        """Projeta do manifold de volta ao espaco de embeddings.

        Args:
            coords: [B, T, d_manifold]

        Returns:
            Tensor [B, T, d_model].
        """
        return self.inv_proj(coords)


def gamma_scale(
    coords: torch.Tensor,
    anchor_coords: torch.Tensor,
    c_param: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fator de Lorentz baseado na distancia ao anchor mais proximo.

    gamma(v) = 1 / sqrt(1 - v^2/c^2), onde v e a distancia Euclidiana
    ao anchor mais proximo e c e a velocidade limite.

    Args:
        coords: [B, T, D] coordenadas dos tokens no manifold.
        anchor_coords: [A, D] coordenadas dos anchors.
        c_param: Velocidade limite.
        eps: Epsilon numerico.

    Returns:
        Tensor [B, T, 1] com fator de escala >= 1.0.
    """
    delta = coords.unsqueeze(2) - anchor_coords.unsqueeze(0).unsqueeze(0)

    dists = delta.norm(dim=-1)

    v = dists.min(dim=-1, keepdim=True).values

    v = v.clamp(max=c_param * 0.999)

    gamma = 1.0 / torch.sqrt(1.0 - (v / c_param) ** 2 + eps)

    return gamma
