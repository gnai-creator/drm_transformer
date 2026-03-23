"""DRM Attention com distancia geodesica low-rank, RoPE e gamma-scaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

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
    """Multi-Head Attention com distancia geodesica low-rank no manifold DRM.

    Usa score(i,j) = -d_G(q_i, k_j) / temp em vez de dot product,
    onde d_G e a distancia sob G(x) = I + U(x) U(x)^T.

    dist^2 = ||delta||^2 + ||U^T delta||^2

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

        # Distancia low-rank: dist^2 = ||delta||^2 + ||U^T delta||^2
        # Complexidade: O(T^2 * D * r)
        delta = q_manifold.unsqueeze(3) - k_manifold.unsqueeze(2)  # [B, H, T, T, D]

        # Parte euclidiana: ||delta||^2
        dist_euc = (delta ** 2).sum(dim=-1)  # [B, H, T, T]

        # Parte low-rank: ||U^T delta||^2
        # U: [B, H, T, D, r] -> U_exp: [B, H, T, 1, D, r]
        U_exp = U.unsqueeze(3)
        delta_col = delta.unsqueeze(-1)  # [B, H, T, T, D, 1]
        Ut_delta = torch.matmul(
            U_exp.transpose(-1, -2), delta_col,
        ).squeeze(-1)  # [B, H, T, T, r]
        dist_lr = (Ut_delta ** 2).sum(dim=-1)  # [B, H, T, T]

        dist_sq = (dist_euc + dist_lr).clamp(min=0.0)

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
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out
