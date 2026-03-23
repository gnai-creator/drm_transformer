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


# Nomes dos anchors semanticos (para logging e interpretabilidade)
ANCHOR_NAMES = [
    "truth",       # baixa incerteza, alta qualidade
    "ignorance",   # alta incerteza epistemica
    "safety",      # regiao de seguranca
    "complexity",  # alta complexidade de dominio
    "creativity",  # exploracao, novidade
    "grounding",   # factualidade, evidencia
]

# Primeiras 4 dims de cada anchor (dims extras preenchidas com 0.5)
_ANCHOR_SEEDS = [
    [0.1, 0.1, 0.5, 0.9],   # truth: baixa incerteza, alta qualidade
    [0.9, 0.9, 0.5, 0.2],   # ignorance: alta incerteza epistemica
    [0.1, 0.5, 0.1, 0.8],   # safety: seguranca, gamma alto longe daqui
    [0.5, 0.5, 0.9, 0.5],   # complexity: alta complexidade
    [0.8, 0.2, 0.5, 0.5],   # creativity: exploracao
    [0.2, 0.2, 0.2, 0.7],   # grounding: factualidade
]


def create_semantic_anchors(
    d_manifold: int,
    n_anchors: int = 6,
) -> torch.Tensor:
    """Cria anchors com posicoes semanticas no manifold.

    Inicializa com significado interpretavel mas mantidos como
    nn.Parameter para o optimizer ajustar durante treino.

    Os primeiros min(4, d_manifold) dims tem valores semanticos;
    dims restantes preenchidas com 0.5 (neutro).

    Se n_anchors > 6, extras inicializados com rand.
    Se n_anchors <= 6, usa os primeiros n_anchors.

    Args:
        d_manifold: Dimensao do manifold.
        n_anchors: Numero de anchors.

    Returns:
        Tensor [n_anchors, d_manifold] em [0, 1].
    """
    anchors = torch.full((n_anchors, d_manifold), 0.5)

    n_seeds = min(n_anchors, len(_ANCHOR_SEEDS))
    seed_dims = min(4, d_manifold)

    for i in range(n_seeds):
        anchors[i, :seed_dims] = torch.tensor(_ANCHOR_SEEDS[i][:seed_dims])

    # Extras aleatorios (se n_anchors > 6)
    if n_anchors > len(_ANCHOR_SEEDS):
        anchors[len(_ANCHOR_SEEDS):] = torch.rand(
            n_anchors - len(_ANCHOR_SEEDS), d_manifold,
        )

    return anchors


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
