"""DRM Attention com distancia Riemanniana low-rank, RoPE e gamma-scaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .config import DRMTransformerConfig
from .metric_net import MetricNet
from .gravity import GravityField
from .manifold import gamma_scale


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding para codificacao de posicao relativa.

    Args:
        d_head: Dimensao por attention head.
        max_seq_len: Comprimento maximo de sequencia.
    """

    def __init__(self, d_head: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, d_head, 2).float() / d_head)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computa cos e sin para RoPE.

        Args:
            x: Tensor de referencia para device.
            seq_len: Comprimento da sequencia.

        Returns:
            Tupla (cos, sin) com shape [seq_len, d_head].
        """
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica rotacao RoPE ao tensor.

    Args:
        x: [B, H, T, d_head] tensor de queries ou keys.
        cos: [T, d_head] componente cosseno.
        sin: [T, d_head] componente seno.

    Returns:
        Tensor [B, H, T, d_head] rotacionado.
    """
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    c1, s1 = cos[..., :half], sin[..., :half]
    return torch.cat([x1 * c1 - x2 * s1, x1 * s1 + x2 * c1], dim=-1)


class DRMAttention(nn.Module):
    """Multi-Head Attention com distancia Riemanniana low-rank no manifold DRM.

    Usa score(i,j) = -d_G(q_i, k_j) / temp em vez de dot product,
    onde d_G e uma distancia local ou aproximacao por quadratura sob
    G(x) = I + U(x) U(x)^T.

    Modo local:
        dist^2 = ||delta||^2 + ||U(q)^T delta||^2

    Modo quadrature:
        aproxima o comprimento do segmento q-k integrando
        sqrt(dx^T G(x(t)) dx), com x(t) = (1-t)q + tk.

    Complexidade: O(T^2 * D * r) onde r e o rank (tipicamente 4).

    Args:
        config: Configuracao do DRM Transformer.
    """

    def __init__(self, config: DRMTransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        self.d_manifold = config.d_manifold
        self.gamma_enabled = config.gamma_enabled
        self.gamma_c = config.gamma_c
        self.gamma_alpha = getattr(config, "gamma_alpha", 0.0)
        self.distance_mode = getattr(config, "distance_mode", "local")
        self.quad_points = int(getattr(config, "quad_points", 0) or getattr(config, "n_quad", 0))
        if self.distance_mode == "quadrature" and self.quad_points <= 0:
            self.distance_mode = "local"
        if self.distance_mode not in ("local", "quadrature"):
            raise ValueError(f"distance_mode must be 'local' or 'quadrature', got {self.distance_mode!r}")
        self.distance_chunk_size = int(getattr(config, "distance_chunk_size", 0) or 0)
        self.last_distance_diagnostics: Dict[str, torch.Tensor] = {}
        self.last_distance_components: Dict[str, torch.Tensor] = {}
        self.last_attention: Optional[torch.Tensor] = None

        assert config.d_model % config.n_heads == 0

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.q_to_manifold = nn.Linear(self.d_head, config.d_manifold, bias=False)
        self.k_to_manifold = nn.Linear(self.d_head, config.d_manifold, bias=False)

        self.temperature = nn.Parameter(
            torch.tensor(getattr(config, "temperature_init", 1.0))
        )
        self.temperature_min = getattr(config, "temperature_min", 0.5)

        self.rope = RotaryEmbedding(self.d_head, config.max_seq_len)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _quadrature_grid(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna pontos/pesos midpoint em [0, 1] para integracao estavel."""
        n = max(int(self.quad_points), 1)
        t = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
        w = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        return t, w

    def _local_distance(
        self,
        q_manifold: torch.Tensor,
        k_manifold: torch.Tensor,
        U: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Distancia local low-rank avaliada em G(q)."""
        delta = q_manifold.unsqueeze(3) - k_manifold.unsqueeze(2)
        dist_euc = delta.pow(2).sum(dim=-1)
        U_exp = U.unsqueeze(3)
        delta_col = delta.unsqueeze(-1)
        Ut_delta = torch.matmul(U_exp.transpose(-1, -2), delta_col).squeeze(-1)
        dist_lr = Ut_delta.pow(2).sum(dim=-1)
        dist_sq = (dist_euc + dist_lr).clamp(min=0.0)
        return dist_sq, dist_euc, dist_lr

    def _quadrature_distance(
        self,
        q_manifold: torch.Tensor,
        k_manifold: torch.Tensor,
        metric_net: MetricNet,
        gravity_field: Optional[GravityField] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aproxima comprimento Riemanniano no segmento q-k por quadratura.

        Retorna ``length^2`` para manter a escala do modo local usada pelo
        softmax da atencao.
        """
        B, H, T, D = q_manifold.shape
        chunk = self.distance_chunk_size if self.distance_chunk_size > 0 else T
        t_grid, w_grid = self._quadrature_grid(q_manifold.device, q_manifold.dtype)

        dist_sq_parts = []
        dist_euc_parts = []
        dist_lr_parts = []
        k_all = k_manifold.unsqueeze(2)

        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            q_chunk = q_manifold[:, :, start:end, :]
            delta = q_chunk.unsqueeze(3) - k_all
            dist_euc = delta.pow(2).sum(dim=-1)
            delta_for_path = -delta

            weighted_length = torch.zeros_like(dist_euc)
            weighted_lr = torch.zeros_like(dist_euc)
            for t, w in zip(t_grid, w_grid):
                x_t = (1.0 - t) * q_chunk.unsqueeze(3) + t * k_all
                U_t = metric_net(x_t.reshape(-1, D)).view(B, H, end - start, T, D, metric_net.rank)
                if gravity_field is not None:
                    U_grav = []
                    for h in range(H):
                        coords_h = x_t[:, h].reshape(B, -1, D)
                        U_h = U_t[:, h].reshape(B, -1, D, metric_net.rank)
                        mass_h = gravity_field.compute_mass(coords_h)
                        U_h = gravity_field.deform_U(U_h, coords_h, mass_h)
                        U_grav.append(U_h.view(B, end - start, T, D, metric_net.rank))
                    U_t = torch.stack(U_grav, dim=1)
                Ut_delta = torch.matmul(
                    U_t.transpose(-1, -2),
                    delta_for_path.unsqueeze(-1),
                ).squeeze(-1)
                dist_lr_t = Ut_delta.pow(2).sum(dim=-1)
                integrand = (dist_euc + dist_lr_t).clamp_min(1e-12).sqrt()
                weighted_length = weighted_length + w * integrand
                weighted_lr = weighted_lr + w * dist_lr_t

            dist_sq_parts.append(weighted_length.pow(2).clamp(min=0.0))
            dist_euc_parts.append(dist_euc)
            dist_lr_parts.append(weighted_lr.clamp(min=0.0))

        return (
            torch.cat(dist_sq_parts, dim=2),
            torch.cat(dist_euc_parts, dim=2),
            torch.cat(dist_lr_parts, dim=2),
        )

    @staticmethod
    def _distance_diagnostics(
        U: torch.Tensor,
        dist_euc: torch.Tensor,
        dist_lr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Metricas compactas para detectar colapso ou efeito real de U."""
        eps = torch.as_tensor(1e-8, device=dist_euc.device, dtype=dist_euc.dtype)
        U_norm = U.pow(2).sum(dim=-2).sqrt()
        dist_delta = dist_lr
        return {
            "metric_U_norm_mean": U_norm.mean().detach(),
            "metric_U_norm_std": U_norm.std(unbiased=False).detach(),
            "metric_U_variance": U.var(unbiased=False).detach(),
            "metric_condition_proxy": (1.0 + U_norm.pow(2).amax()).detach(),
            "geodesic_vs_euclidean_delta_mean": dist_delta.mean().detach(),
            "geodesic_vs_euclidean_delta_std": dist_delta.std(unbiased=False).detach(),
            "dist_lr_fraction": (dist_lr / (dist_euc + dist_lr + eps)).mean().detach(),
        }

    def forward(
        self,
        x: torch.Tensor,
        metric_net: MetricNet,
        gravity_field: Optional[GravityField] = None,
        anchor_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass do DRM Attention.

        Args:
            x: [B, T, d_model] embeddings.
            metric_net: MetricNet para computar U(x) low-rank.
            gravity_field: GravityField opcional para deformar metrica.
            anchor_coords: [A, d_manifold] anchors para gamma-scaling.

        Returns:
            Tensor [B, T, d_model] resultado da attention.
        """
        B, T, C = x.shape
        D = self.d_manifold
        H = self.n_heads
        r = metric_net.rank

        q = self.q_proj(x).view(B, T, H, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.d_head).transpose(1, 2)

        cos, sin = self.rope(q, T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q_manifold = torch.sigmoid(self.q_to_manifold(q))
        k_manifold = torch.sigmoid(self.k_to_manifold(k))

        # Low-rank U(x): [B*H*T, D, r] -> [B, H, T, D, r]
        q_flat = q_manifold.reshape(-1, D)
        U = metric_net(q_flat)
        U = U.view(B, H, T, D, r)

        # Gravidade per-head (modifica U via scaling)
        if gravity_field is not None:
            U_heads = []
            for h in range(H):
                q_h = q_manifold[:, h]  # [B, T, D]
                mass_h = gravity_field.compute_mass(q_h)  # [B, T, 1]
                U_h = gravity_field.deform_U(U[:, h], q_h, mass_h)
                U_heads.append(U_h)
            U = torch.stack(U_heads, dim=1)

        if self.distance_mode == "quadrature":
            dist_sq, dist_euc, dist_lr = self._quadrature_distance(
                q_manifold, k_manifold, metric_net, gravity_field,
            )
        else:
            dist_sq, dist_euc, dist_lr = self._local_distance(q_manifold, k_manifold, U)
        self.last_distance_diagnostics = self._distance_diagnostics(U, dist_euc, dist_lr)
        self.last_distance_components = {
            "dist_sq": dist_sq.detach(),
            "dist_euc": dist_euc.detach(),
            "dist_lr": dist_lr.detach(),
        }

        # Gamma-scaling com log-gamma + annealing + clamp
        if self.gamma_enabled and anchor_coords is not None:
            gamma = gamma_scale(
                q_manifold[:, 0],
                anchor_coords,
                c_param=self.gamma_c,
            )  # [B, T, 1]

            # Normalizacao adaptativa de distancia antes do gamma
            dist_mean = dist_sq.detach().mean(dim=-1, keepdim=True) + 1e-6
            dist_sq = dist_sq / dist_mean

            # Log-gamma suavizado com annealing
            gamma = gamma.clamp(max=3.0)
            gamma_log = torch.log1p(gamma - 1.0)
            alpha = self.gamma_alpha
            effective_gamma = 1.0 + alpha * gamma_log  # [B, T, 1]

            # Broadcast: [B, T, 1] -> [B, 1, T, 1] para [B, H, T, T]
            effective_gamma_sq = (effective_gamma ** 2).unsqueeze(1)
            dist_sq = dist_sq * effective_gamma_sq

        temp = self.temperature.clamp(min=self.temperature_min)
        attn = -dist_sq / temp

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        self.last_attention = attn.detach()
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out
