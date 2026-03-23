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
Geodesic Attention com tensor metrico aprendido G(x), campo gravitacional
por token e dimensionalidade variavel.

## Dados de Treino

| Campo | Valor |
|-------|-------|
| Fonte | CulturaX (uonlp/CulturaX) |
| Linguas | en, pt, es, fr, de |
| Tokens alvo | 20B |
| Tokenizer | tiktoken o200k_base (remapeado para 50K) |
| Baseline | Wikipedia EN 10M tokens (publico) |

## Configuracoes Disponíveis

12 scaling configs de 1M a 640B parametros. Baseline canonico: small_1m (1M params).

## Metricas (Baseline small_1m, seed=42, 10M tokens Wikipedia EN)

| Metrica | full | no_gravity | no_gamma | no_variable_dim |
|---------|------|------------|----------|-----------------|
| Val Loss | 6.415 | 6.385 | **6.321** | 6.428 |
| Val PPL | 611.2 | 593.2 | **556.1** | 619.2 |
| Train Loss | 6.344 | 5.883 | 6.252 | 6.308 |
| Tokens/s | 86,163 | 92,464 | 86,468 | 86,360 |
| Steps | 2,441 | 2,441 | 2,441 | 2,441 |
| Skip Grads | 0 | 0 | 0 | 0 |
| Tempo | 116s | 108s | 116s | 116s |

**Observacoes (baseline 1M params — resultados preliminares):**
- `no_gamma` obteve melhor val PPL (556.1), sugerindo que gamma-scaling
  pode precisar de mais dados/escala para ser benefico.
- `no_gravity` foi a variante mais rapida (92K tok/s) por evitar o compute RFF.
- Nenhuma variante teve gradient skips, indicando estabilidade numerica.
- Em escala tao pequena (1M params, 10M tokens), as diferencas entre variantes
  sao marginais. Resultados em escala (350M+) pendentes.

## Limitacoes

- **Escala atual**: baseline testado com 1M params / 10M tokens. Resultados
  em escala (350M+) ainda em andamento.
- **Benchmarks**: HellaSwag, ARC e MMLU pendentes — requerem modelo em escala.
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

## Riscos

- Modelo de linguagem generativo pode produzir conteudo incorreto ou prejudicial
- Treinado em dados web (CulturaX = mC4 + OSCAR) que podem conter bias
- Nao passou por RLHF ou alinhamento — saida bruta do pre-treino

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
  title = {DRM Transformer: Decoder-only Transformer with Geodesic Attention},
  year = {2026},
  url = {https://github.com/gnai-creator/drm_transformer},
}
```

---

(c) 2026 Felipe Maya Muniz. All rights reserved.
