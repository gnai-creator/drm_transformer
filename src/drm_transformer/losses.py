"""Losses de regularizacao geometrica para o tensor metrico low-rank."""

import torch
import torch.nn.functional as F


def metric_regularization(U: torch.Tensor) -> torch.Tensor:
    """Regularizacao do fator low-rank U(x).

    Penaliza norma excessiva dos eixos semanticos para manter
    G(x) = I + U U^T proximo da identidade no inicio do treino.

    Aceita U [..., D, r] (low-rank), [..., D] (diagonal legacy),
    ou [..., D, D] (full matrix legacy).

    Args:
        U: Tensor metrico em qualquer formato.

    Returns:
        Loss escalar.
    """
    # Backward compat: full matrix -> extrair diagonal
    if U.dim() >= 2 and U.shape[-1] == U.shape[-2]:
        G_diag = U.diagonal(dim1=-2, dim2=-1)
        G_flat = G_diag.reshape(-1, G_diag.shape[-1])
        scale = ((G_flat - 1.0) ** 2).mean()
        condition = G_flat.var(dim=-1).mean() / (G_flat.mean(dim=-1).pow(2).mean() + 1e-8)
        return condition + 0.1 * scale

    # Diagonal legacy: [..., D]
    if U.dim() >= 1 and (U.dim() < 2 or U.shape[-2] != U.shape[-1]):
        if U.dim() <= 2:
            G_flat = U.reshape(-1, U.shape[-1])
            scale = ((G_flat - 1.0) ** 2).mean()
            condition = G_flat.var(dim=-1).mean() / (G_flat.mean(dim=-1).pow(2).mean() + 1e-8)
            return condition + 0.1 * scale

    # Low-rank U: [..., D, r] — penalizar norma total dos eixos
    # ||U||_F^2 controla quanto G se afasta de I
    U_flat = U.reshape(-1, U.shape[-2], U.shape[-1])
    return U_flat.pow(2).sum(dim=(-2, -1)).mean()


def metric_diversity_loss(
    U: torch.Tensor,
    target_var: float = 0.001,
) -> torch.Tensor:
    """Penaliza U(x) com variancia entre tokens longe do alvo.

    Aceita U [..., D, r] (low-rank) ou [..., D] (diagonal legacy).

    Args:
        U: Tensor metrico.
        target_var: Variancia alvo (default 0.001).

    Returns:
        Loss escalar.
    """
    # Backward compat: full matrix -> diagonal
    if U.dim() >= 2 and U.shape[-1] == U.shape[-2]:
        U = U.diagonal(dim1=-2, dim2=-1)

    if U.dim() <= 1:
        return torch.tensor(0.0, device=U.device)

    if U.dim() == 2:
        # [T, D] ou [T, D*r] — nao tem batch dim
        return torch.tensor(0.0, device=U.device)

    # Low-rank [B, T, D, r]: flatten D*r e variar sobre tokens
    if U.dim() == 4:
        B, T, D, r = U.shape
        U_flat = U.reshape(B, T, D * r)
        var = U_flat.var(dim=1).mean()
        return (var - target_var).pow(2)

    # Diagonal [B, T, D]: variar sobre tokens
    var = U.var(dim=-2).mean()
    return (var - target_var).pow(2)


def orthogonality_loss(U: torch.Tensor) -> torch.Tensor:
    """Regulariza U para eixos semanticos ortogonais: U^T U ~ I.

    Previne colapso de eixos redundantes, forcando cada coluna
    de U a representar uma direcao semantica distinta.

    Args:
        U: [..., D, r] fator low-rank do tensor metrico.

    Returns:
        Loss escalar.
    """
    # U^T U: [..., r, r]
    UtU = torch.matmul(U.transpose(-1, -2), U)
    I = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype)
    return ((UtU - I) ** 2).mean()


def axis_variance_loss(U: torch.Tensor) -> torch.Tensor:
    """Penaliza eixos semanticos constantes entre tokens.

    Encoraja U(x) a variar ao longo da sequencia, de modo que
    a curvatura do manifold seja dependente de posicao.

    Args:
        U: [B, H, T, D, r] ou [B, T, D, r] fator low-rank.

    Returns:
        Loss escalar (negativo da variancia, para minimizar).
    """
    # Variancia sobre a dimensao de tokens (dim=2 para [B,H,T,D,r])
    if U.dim() == 5:
        return -U.var(dim=2).mean()
    if U.dim() == 4:
        return -U.var(dim=1).mean()
    return torch.tensor(0.0, device=U.device)


def anchor_alignment_loss(
    U: torch.Tensor,
    coords: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Alinhamento suave dos eixos semanticos com anchors.

    Para cada token, identifica o anchor mais proximo e encoraja
    o primeiro eixo de U a apontar na direcao desse anchor.
    Alinhamento e suave (cosine similarity), sem hard constraints.

    Args:
        U: [B, H, T, D, r] fator low-rank.
        coords: [B, H, T, D] coordenadas no manifold.
        anchors: [A, D] coordenadas dos anchors semanticos.

    Returns:
        Loss escalar (negativo do alinhamento medio).
    """
    # Usar primeiro head para alinhamento (anchors sao compartilhados)
    if coords.dim() == 4:
        coords_flat = coords[:, 0]  # [B, T, D]
        U_flat = U[:, 0]  # [B, T, D, r]
    else:
        coords_flat = coords  # [B, T, D]
        U_flat = U  # [B, T, D, r]

    # Distancia ao anchor mais proximo
    dist = torch.cdist(coords_flat, anchors.unsqueeze(0).expand(coords_flat.shape[0], -1, -1))
    closest_idx = dist.argmin(dim=-1)  # [B, T]

    # Primeiro eixo normalizado
    axis0 = F.normalize(U_flat[..., 0], dim=-1)  # [B, T, D]

    # Direcao do anchor mais proximo
    anchor_vecs = anchors[closest_idx]  # [B, T, D]
    anchor_vecs = F.normalize(anchor_vecs, dim=-1)

    # Cosine similarity suave
    alignment = (axis0 * anchor_vecs).sum(dim=-1)  # [B, T]

    return -alignment.mean()
