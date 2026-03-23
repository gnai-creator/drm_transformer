# 001 - Voronoi Foliation Pipeline para DRM Transformer

Data: 2026-03-22

## Contexto

Pipeline de validacao topologica adaptado do aletheion-llm-v2 para o
drm-transformer. Permite computar H1/H2 (homologia persistente) e ARI
(estabilidade de Voronoi) sobre as representacoes do manifold DRM.

## Diferencas vs Aletheion-LLM-v2

- Nao ha epistemic head separada — coords extraidas via q_to_manifold do layer 0
- d_manifold variavel (4-40), nao fixo em 5
- Sem q1/q2/confidence/vi/phi — usa coords raw + G_diag + gamma + mass
- G(x) sempre disponivel (MetricNet e parte do modelo)
- Anchors sao nn.Parameter (4 pontos), nao fixos
- Reeb graph usa gamma/mass como funcao de Morse (nao confidence/phi)

## Arquivos Criados

### scripts/extract_drm_vectors.py
- Carrega checkpoint e dataset
- Forward pass parcial: embeddings -> dim_gate -> block0 q_to_manifold
- Extrai: coords [N, D], G_diag [N, D], gamma [N, 1], mass [N, 1]
- Salva .npy + metadata.json

### scripts/voronoi_foliation_drm.py
9 fases:
1. Voronoi Tessellation (KMeans)
2. Local Tangent Space Analysis (PCA por celula)
3. Tangent Coherence (angulo entre espacos tangentes)
4. Reeb Graph (level sets de gamma/mass/coord_0)
5. Persistent Homology (H0, H1, H2 via ripser)
6. Null Models (shuffled, uniform)
7. Stability (ARI entre re-runs)
8. Foliation Score (F = (1-H/Hmax) * coherence * ARI)
9. Coords Correlation (ANOVA F-stat por eixo)

### src/drm_transformer/evaluation/foliation.py
- DRMFoliationEvaluator: pipeline completo em classe Python
- extract_vectors() + compute_foliation() + evaluate()

## Dependencias Adicionadas

pyproject.toml [eval]: ripser, scikit-learn, scipy, numpy

## Atualizacoes

- README.md: secao Voronoi Foliation + API Python + estrutura atualizada
- pyproject.toml: extra [eval]
