"""Losses de regularizacao geometrica para o tensor metrico."""

import torch


def metric_regularization(G: torch.Tensor) -> torch.Tensor:
    """Regularizacao do tensor metrico G(x).

    Penaliza G longe da identidade (condition number alto)
    e escala anormal (trace longe de dim).

    Args:
        G: [B, T, D, D] ou [D, D] tensor metrico.

    Returns:
        Loss escalar.
    """
    if G.dim() == 2:
        D = G.shape[0]
        frob_sq = G.pow(2).sum()
        trace = G.diagonal().sum()
        condition = (frob_sq / (trace ** 2 + 1e-8) - 1.0 / D).clamp(min=0)
        scale = (trace / D - 1.0) ** 2
        return condition + 0.1 * scale

    D = G.shape[-1]
    G_flat = G.reshape(-1, D, D)
    diag = G_flat.diagonal(dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    frob_sq = G_flat.pow(2).sum(dim=(-1, -2))
    condition = (frob_sq / (trace ** 2 + 1e-8) - 1.0 / D).clamp(min=0).mean()
    scale = ((trace / D - 1.0) ** 2).mean()
    return condition + 0.1 * scale


def metric_diversity_loss(G: torch.Tensor) -> torch.Tensor:
    """Penaliza G(x) constante entre posicoes.

    Forca variancia espacial da metrica: -log(var(G) + eps),
    evitando espaco plano sem curvatura aprendida.

    Args:
        G: [B, T, D, D] tensor metrico.

    Returns:
        Loss escalar.
    """
    if G.dim() == 2:
        return torch.tensor(0.0, device=G.device)
    B, T, D, _ = G.shape
    G_flat = G.reshape(B, T, D * D)
    var = G_flat.var(dim=1).mean()
    return (-torch.log(var + 1e-4)).clamp(min=0)
