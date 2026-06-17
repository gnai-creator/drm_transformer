"""Tensor metrico low-rank aprendido G(x) = I + U(x) U(x)^T."""

import torch
import torch.nn as nn


class MetricNet(nn.Module):
    """Computa fator low-rank U(x) do tensor metrico G(x) = I + U U^T.

    Cada coluna de U(x) representa um eixo geometrico de curvatura. Esses eixos
    podem ser comparados a anchors interpretaveis, mas nao sao semantica
    supervisionada por si so. A metrica resultante e SPD por construcao
    (I + U U^T tem autovalores >= 1).

    Complexidade: O(D * r) por ponto, vs O(D^2) para metrica full.

    Args:
        dim: Dimensao do manifold.
        rank: Numero de eixos semanticos (colunas de U).
        hidden: Dimensao da camada oculta do MLP.
    """

    def __init__(self, dim: int, rank: int = 4, hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim * rank),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Inicializa U(x) perto de zero, mas com gradiente nao nulo."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

        # Se U fosse exatamente zero, o termo ||U^T delta||^2 teria gradiente
        # nulo em relacao a U. Um std pequeno mantem G(x) perto de I e treinavel.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Computa U(x) para cada ponto no manifold.

        Com pesos inicializados em zero, U(x) = 0 e G(x) = I no inicio,
        garantindo treino estavel desde o primeiro step.

        Args:
            coords: [..., dim] coordenadas no manifold.

        Returns:
            Tensor [..., dim, rank] com eixos geometricos de curvatura.
        """
        raw = self.net(coords)

        if torch.isnan(raw).any():
            batch_shape = coords.shape[:-1]
            return torch.zeros(
                *batch_shape, self.dim, self.rank,
                device=coords.device, dtype=coords.dtype,
            )

        return raw.view(*coords.shape[:-1], self.dim, self.rank)
