"""Modelo principal DRM Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import DRMTransformerConfig
from .metric_net import MetricNet
from .gravity import GravityField
from .dimensional_gate import DimensionalGate
from .layers import RMSNorm, DRMTransformerBlock
from .manifold import create_semantic_anchors


class DRMTransformer(nn.Module):
    """DRM Transformer: decoder-only com geometria geodesica e gravidade.

    Attention usa distancia geodesica sob G(x) aprendida em vez de
    dot product Euclidiano. Tokens com alta informacao criam curvatura
    gravitacional que afeta a attention de outros tokens.

    Args:
        config: Configuracao do DRM Transformer.
    """

    def __init__(self, config: DRMTransformerConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.metric_net = MetricNet(config.d_manifold, config.metric_hidden)

        self.gravity_field = (
            GravityField(
                config.d_manifold,
                config.gravity_strength,
                n_rff=config.gravity_n_rff,
            )
            if config.gravity_enabled else None
        )

        self.dim_gate = (
            DimensionalGate(config.d_model)
            if config.variable_dim else None
        )

        self.anchors = nn.Parameter(
            create_semantic_anchors(config.d_manifold, config.n_anchors),
        )

        self.blocks = nn.ModuleList([
            DRMTransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Inicializa pesos com distribuicao normal (std=0.02)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass completo do DRM Transformer.

        Args:
            input_ids: [B, T] IDs dos tokens.
            targets: [B, T] IDs alvo para calculo de loss.

        Returns:
            Tupla (logits, loss) onde loss e None se targets nao fornecido.
        """
        B, T = input_ids.shape

        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)

        dimD = None
        if self.dim_gate is not None:
            x, dimD = self.dim_gate(x)

        for block in self.blocks:
            x = block(
                x,
                metric_net=self.metric_net,
                gravity_field=self.gravity_field,
                anchor_coords=self.anchors,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Gera texto autoregressivamente.

        Args:
            input_ids: [B, T] tokens de contexto.
            max_new_tokens: Numero maximo de tokens a gerar.
            temperature: Temperatura para amostragem.
            top_k: Top-k filtering.

        Returns:
            Tensor [B, T + max_new_tokens] com tokens gerados.
        """
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
