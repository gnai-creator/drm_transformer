# 002 - Diagonal Metric + Gravity Per-Head + Gamma Stabilization

**Data:** 2026-03-23

## Problema

O DRM Transformer tinha 3 problemas de escalabilidade:

1. **O(T^2 * d^2)**: MetricNet produzia tensor metrico full `[D, D]`. A computacao de distancia
   Mahalanobis expandia G para `[B, H, T, T, D, D]`, inviabilizando contexto longo e modelos grandes.

2. **Gravity so head 0**: `compute_mass(q_manifold[:, 0])` usava apenas head 0 para calcular massa
   gravitacional, criando bottleneck arbitrario sem justificativa teorica.

3. **Gamma explosion**: Fator de Lorentz `gamma = 1/sqrt(1 - v^2/c^2)` podia chegar a ~22x
   (gamma^2 ~500x), saturando softmax e explodindo gradientes.

## Solucao

### 1. Metrica Diagonal — O(T^2 * d)

**metric_net.py**: MetricNet agora produz apenas diagonal `[N, D]` via `softplus(mlp(x)) + 1e-5`.
Removido Cholesky (L, tril_indices). Positividade garantida por softplus.

**attention.py**: Distancia Mahalanobis diagonal:
```python
dist_sq = (delta ** 2 * G_diag.unsqueeze(3)).sum(dim=-1)
```
Elimina `G_expanded [B,H,T,T,D,D]` e `matmul`, reduzindo memoria e FLOPS por fator ~d.

### 2. Gravity Per-Head

**attention.py**: Massa computada independentemente por head:
```python
for h in range(n_heads):
    mass_h = gravity_field.compute_mass(q_manifold[:, h])
    G_diag_heads.append(gravity_field.deform_metric_diag(...))
G_diag = torch.stack(G_diag_heads, dim=1)
```

**gravity.py**: Novo metodo `deform_metric_diag(G_diag, coords, mass)` que opera em `[B, T, D]`.
Logica RFF extraida para `_compute_rff_influence()` compartilhado. `deform_metric` (full matrix)
mantido para backward compatibility.

### 3. Gamma Estabilizado

**manifold.py**: `gamma_scale()` agora aplica `gamma.clamp(max=3.0)` (gamma^2 max = 9x).

**attention.py**: Tres camadas de protecao:
1. Normalizacao adaptativa de distancia: `dist_sq / dist_sq.detach().mean()`
2. Log-gamma suavizado: `log1p(gamma - 1)` comprime escala nao-linearmente
3. Annealing via `gamma_alpha` (config): comeca em 0 (sem gamma), cresce para 1 durante warmup

**config.py**: Adicionado `gamma_alpha: float = 0.0`.

## Arquivos Modificados

| Arquivo | Mudanca |
|---------|---------|
| `src/drm_transformer/metric_net.py` | Diagonal output, removido Cholesky |
| `src/drm_transformer/attention.py` | Distancia diagonal, gravity per-head, gamma estavel |
| `src/drm_transformer/gravity.py` | `deform_metric_diag`, `_compute_rff_influence` |
| `src/drm_transformer/manifold.py` | Clamp gamma max=3.0 |
| `src/drm_transformer/config.py` | `gamma_alpha` field |
| `src/drm_transformer/losses.py` | Suporte diagonal `[B,T,D]` + backward compat full matrix |
| `src/drm_transformer/training/trainer.py` | Adaptado para diagonal (2 callsites) |
| `scripts/extract_drm_vectors.py` | Removido `.diagonal()` |
| `src/drm_transformer/evaluation/foliation.py` | Removido `.diagonal()` |

## Impacto

- **Complexidade**: O(T^2 * d) vs O(T^2 * d^2) — ~16x menos com d=16
- **Estabilidade**: gamma max 3.0, annealing gradual, normalizacao de distancia
- **Corretude**: cada head tem paisagem gravitacional independente
- **Backward compat**: API externa inalterada, losses aceitam ambos formatos

## Proximos Passos

- Treinar com `gamma_alpha=0.0`, fazer warmup linear para 1.0 nos primeiros 10% dos steps
- Se diagonal perder expressividade: considerar low-rank G = I + U U^T (rank 3-4)
- Benchmark de throughput com T=1024 vs versao anterior
