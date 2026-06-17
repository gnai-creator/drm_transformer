# Model Card: DRM Transformer

## Visao Geral

| Campo | Valor |
|-------|-------|
| Nome | DRM Transformer |
| Tipo | Decoder-only language model |
| Autor | Felipe Maya Muniz |
| Licenca | AGPL-3.0 + Comercial |
| Repositorio | github.com/gnai-creator/drm_transformer |

## Descricao

Transformer decoder-only onde o espaco de embeddings vive num Directional
Relational Manifold (DRM). A atencao padrao (dot-product) e substituida por
Low-Rank Riemannian Attention com tensor metrico aprendido
`G(x)=I+U(x)U(x)^T`, campo gravitacional por token e dimensionalidade variavel.
O modo default usa uma distancia local tipo Mahalanobis; o modo opcional de
quadratura aproxima o comprimento do segmento q-k. O projeto e experimental e
nao valida uma geodesica formal plena nem alinhamento/safety definitivos.

## Dados de Treino

| Campo | Valor |
|-------|-------|
| Fonte | CulturaX (uonlp/CulturaX) |
| Linguas | en, pt, es, fr, de |
| Tokens alvo | 20B |
| Tokenizer | tiktoken o200k_base (remapeado para 50K) |
| Baseline | Wikipedia EN 10M tokens (publico) |

## Configuracoes Disponiveis

12 scaling configs de 1M a 640B parametros. Baseline canonico: small_3.5M (3.5M params).

## Current Empirical Status

O status atual e implementavel e experimental. Os resultados abaixo sao de um
baseline pequeno (3.5M parametros, 10M tokens Wikipedia EN), uteis para
sanidade de engenharia e ablations iniciais. Eles nao demonstram superioridade
sobre Transformers padrao nem validam claims cientificos gerais.

## Metricas (Baseline small_3.5M, seed=42, 10M tokens Wikipedia EN)

### Linguagem

| Metrica | full | no_gravity | no_gamma | no_variable_dim |
|---------|------|------------|----------|-----------------|
| Val Loss | 6.415 | 6.385 | **6.321** | 6.428 |
| Val PPL | 611.2 | 593.2 | **556.1** | 619.2 |
| Train Loss | 6.344 | 5.883 | 6.252 | 6.308 |
| Tokens/s | 86,163 | 92,464 | 86,468 | 86,360 |
| Steps | 2,441 | 2,441 | 2,441 | 2,441 |
| Skip Grads | 0 | 0 | 0 | 0 |
| Tempo | 116s | 108s | 116s | 116s |

**Observacoes (baseline 3.5M params -- resultados preliminares):**
- `no_gamma` obteve melhor val PPL (556.1), sugerindo que gamma-scaling
  pode precisar de mais dados/escala para ser benefico.
- `no_gravity` foi a variante mais rapida (92K tok/s) por evitar o compute RFF.
- Nenhuma variante teve gradient skips, indicando estabilidade numerica.
- Em escala tao pequena (3.5M params, 10M tokens), as diferencas entre variantes
  sao marginais. Resultados em escala (350M+) pendentes.

### Topologia DRM

Resultado de Voronoi Foliation no baseline small_3.5M sob regularizacao
toroidal configurada:

| Metrica | Valor |
|---------|-------|
| Topologia | assinatura compativel com T^2 sob regularizacao |
| H1 long bars | 2 |
| H2 long bars | 1 |
| T2 stable fraction | 0.60 |
| Foliation score | 0.4410 |
| ARI | 0.7959 |
| Homology points | 1200 |
| Homology restarts | 5 |
| Long-bar ratio | 0.75 |

Comando de referencia:

```bash
python scripts/voronoi_foliation_drm.py \
    --coords eval-results/foliation_3.5m/drm_coords.npy \
    --G-diag eval-results/foliation_3.5m/drm_G_diag.npy \
    --gamma eval-results/foliation_3.5m/drm_gamma.npy \
    --output-dir eval-results/foliation_3.5m \
    --n-seeds 80 \
    --homology-points 1200 \
    --homology-long-bar-ratio 0.75 \
    --homology-restarts 5
```

O resultado mostra uma assinatura `H1=2, H2=1` estavel pelo criterio
`t2_stable_fraction >= 0.60` nesse regime. Como `lambda_torus > 0` induz
explicitamente estrutura toroidal, isso nao prova emergencia espontanea de um
toro. A recomendacao cientifica e comparar contra `configs/ablations/no_torus.yaml`
e relatar homologia, LTSA e metricas de linguagem nos dois regimes.

## Limitacoes

- **Escala atual**: baseline testado com 3.5M params / 10M tokens. Resultados
  em escala (350M+) ainda em andamento.
- **Geometria topologica**: `H1=2, H2=1` foi observado no baseline pequeno sob
  regularizacao toroidal; isso e inducao por loss quando `lambda_torus > 0`.
- **Benchmarks**: HellaSwag, ARC e MMLU pendentes -- requerem modelo em escala.
- **Anchors**: truth/safety/grounding sao priors geometricos interpretaveis.
  Validacao semantica exige probes rotulados como `scripts/eval_anchor_probe.py`.
- **MetricNet**: diagnosticos de `dist_lr_fraction` e norma de U devem ser
  monitorados para evitar uma geometria quase Euclidiana nao detectada.
- **Linguas**: treinado em 5 linguas europeias. Performance em outras linguas
  nao avaliada.
- **Determinismo multi-GPU**: NCCL pode introduzir nao-determinismo.
  Reprodutibilidade total requer single GPU.

## Uso Recomendado

- **Pesquisa**: investigar efeitos de geometria Riemanniana em transformers
- **Ablacoes**: comparar contribuicao de gravity, gamma-scaling, DimensionalGate
- **Baseline**: validar que o pipeline de treino funciona antes de escalar

## Uso NAO Recomendado

- **Producao**: modelo experimental, nao validado para uso em producao
- **Tarefas criticas**: sem avaliacao de seguranca ou bias sistematico
- **Substituicao de modelos existentes**: nao superou benchmarks padrao (ainda)
- **Alinhamento definitivo**: anchors semanticos e gamma-scaling nao constituem
  prova de alinhamento, factualidade ou seguranca.

## Riscos

- Modelo de linguagem generativo pode produzir conteudo incorreto ou prejudicial
- Treinado em dados web (CulturaX = mC4 + OSCAR) que podem conter bias
- Nao passou por RLHF ou alinhamento -- saida bruta do pre-treino

## Reprodutibilidade

```bash
git clone https://github.com/gnai-creator/drm_transformer.git
cd drm_transformer
pip install -r requirements-lock.txt
python scripts/repro_baseline.py
```

Ver `repro.md` para guia detalhado.

## Citacao

```bibtex
@software{muniz2026drm,
  author = {Muniz, Felipe Maya},
  title = {DRM Transformer: Decoder-only Transformer with Low-Rank Riemannian Attention},
  year = {2026},
  url = {https://github.com/gnai-creator/drm_transformer},
}
```

---

(c) 2026 Felipe Maya Muniz. All rights reserved.
