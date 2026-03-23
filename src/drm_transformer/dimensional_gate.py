"""Gate de dimensionalidade variavel por token."""

import torch
import torch.nn as nn
from typing import Tuple


class DimensionalGate(nn.Module):
    """Controla dimensoes ativas por token (dimD(p)).

    Cada token recebe um gate em [0, 1] por dimensao,
    implementando dimensionalidade variavel conforme DRM Def 3.1.

    Args:
        d_model: Dimensao do modelo.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aplica gate dimensional.

        Args:
            x: [B, T, d_model] embeddings.

        Returns:
            Tupla (gated, dimD) onde gated tem dims mascaradas
            e dimD [B, T, 1] e a dimensionalidade efetiva.
        """
        gate = self.gate_net(x)

        gated = x * gate

        dimD = gate.sum(dim=-1, keepdim=True)

        return gated, dimD
