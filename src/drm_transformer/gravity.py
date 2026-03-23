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

    def _compute_rff_influence(
        self,
        coords: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Computa influencia gravitacional escalar via RFF.

        Args:
            coords: [B, T, D] coordenadas dos tokens.
            mass: [B, T, 1] massa de cada token.

        Returns:
            Tensor [B, T, 1] com influencia gravitacional por token.
        """
        phi = self._rff_features(coords)  # [B, T, R]
        phi_weighted = phi * mass  # [B, T, R]
        phi_sum = phi_weighted.sum(dim=1, keepdim=True)  # [B, 1, R]
        return (phi * phi_sum).sum(dim=-1, keepdim=True)  # [B, T, 1]

    def deform_U(
        self,
        U: torch.Tensor,
        coords: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Deforma fator low-rank U(x) pela gravidade dos tokens via RFF.

        Escala U pela raiz da influencia gravitacional, de modo que
        G_grav = I + U_grav U_grav^T incorpora curvatura gravitacional.
        O fator sqrt garante que a deformacao em G seja proporcional
        a influencia (nao ao quadrado dela).

        Complexidade: O(T * R).

        Args:
            U: [B, T, D, r] fator low-rank da metrica.
            coords: [B, T, D] coordenadas dos tokens.
            mass: [B, T, 1] massa de cada token.

        Returns:
            Tensor [B, T, D, r] com fator low-rank deformado.
        """
        grav_influence = self._compute_rff_influence(coords, mass)  # [B, T, 1]
        # sqrt(1 + s*g) escala U tal que U U^T cresce linearmente com influencia
        # clamp na influencia antes de somar: evita regioes mortas e explosao
        influence = (self.strength * grav_influence).clamp(min=-0.9, max=5.0)
        scale = torch.sqrt(1.0 + influence + 1e-6).unsqueeze(-1)  # [B, T, 1, 1]
        return U * scale

    def deform_metric_diag(
        self,
        G_diag: torch.Tensor,
        coords: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Deforma diagonal de G(x) pela gravidade dos tokens via RFF.

        Mantida para backward compatibility.

        Args:
            G_diag: [B, T, D] diagonal da metrica base.
            coords: [B, T, D] coordenadas dos tokens.
            mass: [B, T, 1] massa de cada token.

        Returns:
            Tensor [B, T, D] com diagonal da metrica deformada.
        """
        grav_influence = self._compute_rff_influence(coords, mass)  # [B, T, 1]
        influence = (self.strength * grav_influence).clamp(min=-0.9, max=5.0)
        return G_diag * (1.0 + influence + 1e-6)

    def deform_metric(
        self,
        G: torch.Tensor,
        coords: torch.Tensor,
        mass: torch.Tensor,
    ) -> torch.Tensor:
        """Deforma G(x) pela gravidade dos tokens via RFF (versao full matrix).

        Mantida para backward compatibility. Prefer deform_metric_diag.

        Args:
            G: [B, T, D, D] metrica base.
            coords: [B, T, D] coordenadas dos tokens.
            mass: [B, T, 1] massa de cada token.

        Returns:
            Tensor [B, T, D, D] com metrica deformada.
        """
        B, T, D = coords.shape
        grav_influence = self._compute_rff_influence(coords, mass)  # [B, T, 1]
        grav_influence = grav_influence.unsqueeze(-1)  # [B, T, 1, 1]

        I = torch.eye(D, device=G.device, dtype=G.dtype)
        return G + self.strength * grav_influence * I
