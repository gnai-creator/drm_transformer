"""Camadas auxiliares: RMSNorm, FeedForward, DRMTransformerBlock."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import DRMTransformerConfig
from .metric_net import MetricNet
from .gravity import GravityField
from .attention import DRMAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        d: Dimensao do input.
        eps: Epsilon para estabilidade numerica.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normaliza por RMS.

        Args:
            x: Tensor de entrada.

        Returns:
            Tensor normalizado.
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class FeedForward(nn.Module):
    """Feed-Forward com SwiGLU.

    Args:
        config: Configuracao do DRM Transformer.
    """

    def __init__(self, config: DRMTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass SwiGLU.

        Args:
            x: [B, T, d_model] embeddings.

        Returns:
            Tensor [B, T, d_model].
        """
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DRMTransformerBlock(nn.Module):
    """Bloco do DRM Transformer com pre-norm e residual connections.

    Usa DRMAttention (distancia geodesica) em vez de attention padrao.

    Args:
        config: Configuracao do DRM Transformer.
    """

    def __init__(self, config: DRMTransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = DRMAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        metric_net: MetricNet,
        gravity_field: Optional[GravityField] = None,
        anchor_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass do bloco.

        Args:
            x: [B, T, d_model] embeddings.
            metric_net: MetricNet para computar G(x).
            gravity_field: GravityField opcional.
            anchor_coords: Anchors para gamma-scaling.

        Returns:
            Tensor [B, T, d_model].
        """
        x = x + self.attn(
            self.norm1(x), metric_net, gravity_field, anchor_coords,
        )
        x = x + self.ffn(self.norm2(x))
        return x
