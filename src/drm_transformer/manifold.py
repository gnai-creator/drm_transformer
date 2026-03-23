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

def create_semantic_anchors(
    d_manifold: int,
    n_anchors: int = 6,
) -> torch.Tensor:
    """Cria anchors com posicoes semanticas no manifold.

    Usa funcoes periodicas que escalam para qualquer d_manifold:
    cada anchor ocupa um "canto" semantico do hipercubo [0,1]^d.
    Mantidos como nn.Parameter para o optimizer ajustar durante treino.

    Padroes geometricos:
    - truth:      rampa crescente (baixa incerteza -> alta qualidade)
    - ignorance:  rampa decrescente (oposto de truth)
    - safety:     alternado baixo-alto (fronteira de seguranca)
    - complexity: sino (pico no centro do espaco)
    - creativity: alternado alto-baixo (oposto de safety)
    - grounding:  baixo uniforme com pico na ultima dim

    Args:
        d_manifold: Dimensao do manifold (4 a 40+).
        n_anchors: Numero de anchors (default 6).

    Returns:
        Tensor [n_anchors, d_manifold] em [0, 1].
    """
    import math

    semantic_fns = [
        lambda i, d: 0.1 + 0.8 * (i / max(d - 1, 1)),                       # truth
        lambda i, d: 0.9 - 0.8 * (i / max(d - 1, 1)),                       # ignorance
        lambda i, d: 0.1 if i % 2 == 0 else 0.9,                            # safety
        lambda i, d: 0.5 + 0.4 * math.sin(math.pi * i / max(d - 1, 1)),     # complexity
        lambda i, d: 0.9 if i % 2 == 0 else 0.1,                            # creativity
        lambda i, d: 0.2 + 0.6 * (1.0 if i == d - 1 else 0.0),              # grounding
    ]

    anchors = torch.zeros(n_anchors, d_manifold)

    n_semantic = min(n_anchors, len(semantic_fns))
    for a in range(n_semantic):
        for i in range(d_manifold):
            anchors[a, i] = semantic_fns[a](i, d_manifold)

    # Extras aleatorios se n_anchors > 6
    if n_anchors > len(semantic_fns):
        anchors[len(semantic_fns):] = torch.rand(
            n_anchors - len(semantic_fns), d_manifold,
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
