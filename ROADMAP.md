# Roadmap

Plano de desenvolvimento do DRM Transformer.

## v0.1.0 (Concluido)

- [x] Arquitetura decoder-only com DRM Attention
- [x] MetricNet: G(x) via MLP low-rank (SPD garantido, diagonal + semantic axes)
- [x] GravityField: massa dos tokens deforma a metrica (RFF-based)
- [x] DimensionalGate: dimensionalidade variavel dimD(p)
- [x] Gamma-Scaling (fator de Lorentz relativistico)
- [x] Loss composta: CE + metric_reg + metric_diversity
- [x] Script de treinamento (single GPU)
- [x] Testes unitarios basicos
- [x] Licenciamento dual (AGPL-3.0 + Comercial)
- [x] Documentacao (ARCHITECTURE.md, README.md)

## v0.2.0 (Concluido)

- [x] Pipeline de dados streaming com checkpoint/resume (CulturaX + Wikipedia)
- [x] 12 scaling configs multilingual (1M a 640B params)
- [x] Treinamento distribuido (DDP + FSDP + mixed precision + gradient checkpointing)
- [x] Visualizacao do tensor metrico G(x) (heatmap, PCA, t-SNE, separation)
- [x] Metricas de diagnostico: calibracao (ECE, MCE, Brier, perplexidade)
- [x] Suite empirica: 5 testes (axis_statistics, axes_projection, calibration, geometry_correlation, semantic_separation)
- [x] Foliation evaluator (Voronoi, LTSA, Homology, Reeb, ARI)
- [x] Eval in-distribution com --eval-shard
- [x] CulturaX como fonte de dados (6.3T tokens, 167 linguas, download rapido)

## v0.3.0 (Em Andamento)

- [ ] Pre-treinamento completo 20B tokens multilingual (en/pt/es/fr/de) via CulturaX
- [x] Infraestrutura de reprodutibilidade (seed, determinismo, run manifest)
- [x] Baseline canonico small_1m com dataset fixo e early stop
- [x] Matriz de ablacoes (full, no_gravity, no_gamma, no_variable_dim)
- [x] Script de avaliacao padronizada (perplexity, futuro: HellaSwag/ARC)
- [x] Script unico de reproducao (repro_baseline.py)
- [x] Model card tecnico (MODEL_CARD.md)
- [x] Smoke tests CI (8 testes: import, forward, config, seed determinism)
- [x] Rodar ablacoes e gerar results_ablations.md
- [x] Preencher MODEL_CARD.md com numeros reais (baseline 1M)
- [x] Training report plot (loss, PPL, LR, throughput)
- [x] Fix: remap O(1) via lookup array (42s/shard -> ~0.1s/shard)
- [x] Tag de release v0.3.0-baseline com artefatos
- [ ] Benchmarks padrao (HellaSwag, ARC, MMLU) pos-treinamento 350M+
- [ ] Analise de curvatura: G(x) constante vs variavel (H0 vs H1)
- [ ] Comparativo formal: DRM Transformer vs GPT-2 / LLaMA
- [ ] Export ONNX para inferencia otimizada

## v0.4.0 (Planejado)

- [ ] Fine-tuning supervisionado (SFT)
- [ ] Treinamento em escala 1.3B+ (multi-GPU validado)
- [ ] Christoffel symbols e curvatura de Ricci como diagnostico
- [ ] Checkpoint intra-lingua para resume mais granular

## v1.0.0 (Longo Prazo)

- [ ] Treinamento em escala 7B+
- [ ] Integracao com frameworks de servico (vLLM, TGI)
- [ ] API publica de inferencia
- [ ] Paper publicado em conferencia revisada por pares
- [ ] Geodesic attention otimizada (CUDA kernels customizados)

---

Este roadmap e indicativo e pode mudar conforme prioridades e recursos.
Sugestoes sao bem-vindas via issues.
