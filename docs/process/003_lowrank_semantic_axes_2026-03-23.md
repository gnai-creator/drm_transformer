# 003 - Low-Rank Metric com Eixos Semanticos

**Data:** 2026-03-23

## Contexto

A metrica diagonal (implementada em 002) eliminou O(d^2) mas perdeu capacidade de
capturar correlacao entre dimensoes do manifold. Low-rank G(x) = I + U(x) U(x)^T
recupera essa expressividade com custo controlado O(D * r), onde r << D.

Bonus: cada coluna de U(x) pode ser interpretada como um eixo semantico de curvatura
(e.g., safety, truth, grounding), permitindo inspecao do que o modelo aprendeu.

## Implementacao

### MetricNet (metric_net.py)
- Saida: U [..., D, r] em vez de diagonal [..., D]
- MLP: dim -> hidden -> hidden -> dim*rank
- Init zeros na ultima camada (G(x) = I no inicio)
- NaN fallback retorna zeros (G = I)

### Distancia Low-Rank (attention.py)
```
dist^2 = ||delta||^2 + ||U^T delta||^2
```
Onde delta = q - k no manifold. Equivale a delta^T (I + U U^T) delta.
Complexidade: O(T^2 * D * r) com r=4 tipicamente.

### Gravidade (gravity.py)
- Novo metodo `deform_U(U, coords, mass)`: escala U por sqrt(1 + s*g)
- Garante que G_grav = I + U_grav U_grav^T cresce linearmente com influencia
- Per-head como antes

### Losses (losses.py)
Tres novas losses para estruturar os eixos semanticos:

1. **orthogonality_loss(U)**: Penaliza U^T U longe de I. Previne eixos redundantes.
   Peso sugerido: lambda_ortho = 0.01

2. **axis_variance_loss(U)**: Encoraja U(x) a variar entre tokens (curvatura
   position-dependent). Peso sugerido: 0.01

3. **anchor_alignment_loss(U, coords, anchors)**: Alinhamento suave do primeiro
   eixo com direcao do anchor mais proximo. Sem hard constraints, so cosine
   similarity. Peso sugerido: 0.005

### Config (config.py)
- Adicionado `metric_rank: int = 4`

### Trainer (trainer.py)
- Metricas de logging: norma media de U, norma por eixo semantico
- orthogonality_loss integrada no _compute_drm_losses

### Downstream (extract_drm_vectors.py, foliation.py)
- Extraem diagonal de G = I + U U^T: `diag_i = 1 + sum_j U[i,j]^2`
- Compativel com pipelines existentes de visualizacao

## Arquivos Modificados

| Arquivo | Mudanca |
|---------|---------|
| `src/drm_transformer/metric_net.py` | Output U [D, r] low-rank |
| `src/drm_transformer/attention.py` | dist = ||delta||^2 + ||U^T delta||^2 |
| `src/drm_transformer/gravity.py` | deform_U com sqrt scaling |
| `src/drm_transformer/losses.py` | orthogonality, axis_variance, anchor_alignment |
| `src/drm_transformer/config.py` | metric_rank field |
| `src/drm_transformer/model.py` | Passa rank ao MetricNet |
| `src/drm_transformer/training/trainer.py` | Logging de eixos + ortho loss |
| `scripts/extract_drm_vectors.py` | G_diag = 1 + U^2.sum() |
| `src/drm_transformer/evaluation/foliation.py` | G_diag = 1 + U^2.sum() |

## Interpretabilidade Esperada

Apos treino, cada coluna de U deve especializar-se:
- Eixo 0: direcao do anchor mais proximo (safety/truth/grounding)
- Eixos 1-3: direcoes complementares ortogonais

Verificacao via logging:
- `metric_axis{i}_norm`: norma media do eixo i (deve ser > 0, crescente)
- Cosine similarity entre eixos (deve tender a 0 com ortho loss)
- Correlacao com anchors (primeiro eixo deve ter alignment > 0.5)

## Complexidade

| Componente | Antes (full) | Diagonal | Low-rank (r=4) |
|------------|-------------|----------|----------------|
| MetricNet output | D^2 | D | D*r |
| Distancia | O(T^2*D^2) | O(T^2*D) | O(T^2*D*r) |
| Expressividade | Full | Diagonal-only | Rank-r correlacoes |
