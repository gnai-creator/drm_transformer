"""Losses de regularizacao geometrica para o tensor metrico low-rank."""

import torch
import torch.nn.functional as F


def metric_regularization(U: torch.Tensor) -> torch.Tensor:
    """Regularizacao do fator low-rank U(x).

    Penaliza norma excessiva dos eixos semanticos para manter
    G(x) = I + U U^T proximo da identidade no inicio do treino.

    Args:
        U: [..., D, r] fator low-rank do tensor metrico.

    Returns:
        Loss escalar: ||U||_F^2 medio.
    """
    U_flat = U.reshape(-1, U.shape[-2], U.shape[-1])
    return U_flat.pow(2).sum(dim=(-2, -1)).mean()


def metric_diversity_loss(
    U: torch.Tensor,
    target_var: float = 0.001,
) -> torch.Tensor:
    """Penaliza U(x) com variancia entre tokens longe do alvo.

    Args:
        U: [B, T, D, r] fator low-rank do tensor metrico.
        target_var: Variancia alvo (default 0.001).

    Returns:
        Loss escalar.
    """
    if U.dim() < 3:
        return torch.tensor(0.0, device=U.device)

    # Low-rank [B, T, D, r]: variancia por eixo sobre tokens
    if U.dim() == 4:
        # var(dim=1) -> [B, D, r], mean(dim=-2) -> [B, r], mean() -> escalar
        var = U.var(dim=1).mean(dim=-2).mean()
        return (var - target_var).pow(2)

    # [B, T, D]: variar sobre tokens
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


def manifold_variance_loss(
    coords: torch.Tensor,
    target_std: float = 0.08,
) -> torch.Tensor:
    """Penaliza colapso das coordenadas do manifold.

    A loss e zero quando cada eixo tem desvio-padrao medio acima do alvo.
    Diferente da diversity em U(x), esta loss deve receber gradiente ate o
    projetor q_to_manifold para abrir a nuvem de coordenadas.

    Args:
        coords: [B, T, D] ou [B, H, T, D] coordenadas no manifold.
        target_std: Desvio-padrao minimo por eixo.

    Returns:
        Loss escalar.
    """
    if coords.dim() == 4:
        flat = coords.transpose(1, 2).reshape(-1, coords.shape[-1])
    else:
        flat = coords.reshape(-1, coords.shape[-1])

    if flat.shape[0] < 2:
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

    std = flat.float().std(dim=0, unbiased=False)
    target = torch.as_tensor(target_std, device=std.device, dtype=std.dtype)
    return F.relu(target - std).pow(2).mean().to(coords.dtype)


def torus_regularization_loss(
    coords: torch.Tensor,
    target_radius: float = 0.35,
    radial_weight: float = 1.0,
    coverage_weight: float = 0.25,
    isotropy_weight: float = 0.5,
    independence_weight: float = 0.5,
    harmonic_weight: float = 0.5,
) -> torch.Tensor:
    """Regularizacao toroidal para d_manifold >= 4.

    Interpreta os quatro primeiros eixos como dois pares circulares em torno
    de 0.5: (x0, x1) e (x2, x3). Incentiva raio constante e cobertura angular
    dos dois ciclos, que e a assinatura geometrica esperada de T^2 em R^4.

    Args:
        coords: [B, T, D] ou [B, H, T, D] em [0, 1].
        target_radius: Raio alvo em torno de 0.5.
        radial_weight: Peso interno para afinar a espessura radial do toro.
        coverage_weight: Peso interno da penalidade de cobertura angular.
        isotropy_weight: Peso interno para evitar ciclos elipticos/colapsados.
        independence_weight: Peso interno para desacoplar os dois ciclos.
        harmonic_weight: Peso interno para evitar cobertura angular bilobada.

    Returns:
        Loss escalar. Retorna zero se D < 4.
    """
    if coords.shape[-1] < 4:
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

    if coords.dim() == 4:
        flat = coords.transpose(1, 2).reshape(-1, coords.shape[-1])
    else:
        flat = coords.reshape(-1, coords.shape[-1])

    xy = flat[:, 0:2].float() - 0.5
    uv = flat[:, 2:4].float() - 0.5

    r1 = torch.linalg.norm(xy, dim=-1)
    r2 = torch.linalg.norm(uv, dim=-1)
    target = torch.as_tensor(target_radius, device=flat.device, dtype=torch.float32)
    radial = (r1 - target).pow(2).mean() + (r2 - target).pow(2).mean()

    eps = 1e-6
    c1 = xy / r1.clamp_min(eps).unsqueeze(-1)
    c2 = uv / r2.clamp_min(eps).unsqueeze(-1)
    # Boa cobertura angular tem media vetorial perto de zero nos dois ciclos.
    coverage = c1.mean(dim=0).pow(2).sum() + c2.mean(dim=0).pow(2).sum()

    # Um toro em R4 precisa de dois ciclos circulares, nao um ciclo forte e um
    # eixo fino. Para cada par, os segundos momentos dos dois eixos devem ser
    # parecidos e a covariancia cruzada deve ficar perto de zero.
    def _pair_isotropy(pair: torch.Tensor) -> torch.Tensor:
        xx = pair[:, 0].pow(2).mean()
        yy = pair[:, 1].pow(2).mean()
        xy = (pair[:, 0] * pair[:, 1]).mean()
        return (xx - yy).pow(2) + xy.pow(2)

    isotropy = _pair_isotropy(xy) + _pair_isotropy(uv)

    # Coverage pela media vetorial zera tambem em dois lobos opostos. O termo
    # de segunda harmonica penaliza esse falso circulo: em uma circunferencia
    # bem coberta, E[cos(2 theta)] e E[sin(2 theta)] ficam perto de zero.
    def _second_harmonic(unit_pair: torch.Tensor) -> torch.Tensor:
        cos_t = unit_pair[:, 0]
        sin_t = unit_pair[:, 1]
        cos_2t = cos_t.pow(2) - sin_t.pow(2)
        sin_2t = 2.0 * cos_t * sin_t
        return cos_2t.mean().pow(2) + sin_2t.mean().pow(2)

    harmonic = _second_harmonic(c1) + _second_harmonic(c2)

    # Evita que os dois angulos variem juntos como uma curva diagonal em T2.
    # Para um produto S1 x S1 bem coberto, a correlacao entre os vetores
    # circulares dos dois pares deve ficar perto de zero.
    cross = c1.unsqueeze(-1) * c2.unsqueeze(-2)  # [N, 2, 2]
    independence = cross.mean(dim=0).pow(2).sum()

    loss = (
        radial_weight * radial
        + coverage_weight * coverage
        + isotropy_weight * isotropy
        + independence_weight * independence
        + harmonic_weight * harmonic
    )
    return loss.to(coords.dtype)


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
