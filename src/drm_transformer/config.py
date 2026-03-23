"""Configuracao do DRM Transformer."""

from dataclasses import dataclass


@dataclass
class DRMTransformerConfig:
    """Parametros do DRM Transformer, incluindo geometria do manifold.

    Attrs:
        vocab_size: Tamanho do vocabulario.
        max_seq_len: Comprimento maximo de sequencia.
        d_model: Dimensao do modelo (embeddings).
        n_layers: Numero de blocos Transformer.
        n_heads: Numero de attention heads.
        d_ff: Dimensao do feed-forward.
        dropout: Taxa de dropout.
        bias: Usar bias nas camadas lineares.
        d_manifold: Dimensao do manifold epistemico.
        metric_hidden: Dimensao oculta do MetricNet.
        n_quad: Pontos de quadratura Gauss-Legendre (0=Mahalanobis local).
        gamma_enabled: Ativar gamma-scaling relativistic.
        gamma_c: Velocidade limite c para gamma-scaling.
        gravity_enabled: Ativar campo gravitacional.
        gravity_strength: Forca da gravidade.
        variable_dim: Ativar dimensionalidade variavel por token.
    """

    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    bias: bool = False

    d_manifold: int = 16
    metric_hidden: int = 64
    n_quad: int = 0

    n_anchors: int = 6

    gamma_enabled: bool = True
    gamma_c: float = 4.0

    temperature_init: float = 1.0
    temperature_min: float = 0.5

    gravity_enabled: bool = True
    gravity_strength: float = 0.1
    gravity_n_rff: int = 64

    variable_dim: bool = True
