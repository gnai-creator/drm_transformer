# Roadmap

Plano de desenvolvimento do DRM Transformer.

## v0.1.0 (Atual)

- [x] Arquitetura decoder-only com DRM Attention
- [x] MetricNet: G(x) via MLP + Cholesky (SPD garantido)
- [x] GravityField: massa dos tokens deforma a metrica
- [x] DimensionalGate: dimensionalidade variavel dimD(p)
- [x] Gamma-Scaling (fator de Lorentz relativístico)
- [x] Loss composta: CE + metric_reg + metric_diversity
- [x] Script de treinamento (single GPU)
- [x] Testes unitarios basicos
- [x] Licenciamento dual (AGPL-3.0 + Comercial)
- [x] Documentacao (ARCHITECTURE.md, README.md)

## v0.2.0 (Planejado)

- [ ] Pre-treinamento completo em FineWeb-Edu (350M params)
- [ ] Benchmarks padrao (HellaSwag, ARC, MMLU) pos-treinamento
- [ ] Analise de curvatura: G(x) constante vs variavel (H0 vs H1)
- [ ] Visualizacao do tensor metrico G(x) ao longo da sequencia
- [ ] Metricas de diagnostico (condition number, variancia de G)
- [ ] Export ONNX para inferencia otimizada

## v0.3.0 (Futuro)

- [ ] Comparativo formal: DRM Transformer vs GPT-2 / LLaMA
  com foco em qualidade de representacao e perplexidade
- [ ] Ablation studies: gravidade, DimensionalGate, gamma-scaling
- [ ] Fine-tuning supervisionado (SFT)
- [ ] Treinamento em escala 1.3B+ (multi-GPU validado)
- [ ] Christoffel symbols e curvatura de Ricci como diagnostico

## v1.0.0 (Longo Prazo)

- [ ] Treinamento em escala 7B+
- [ ] Integracao com frameworks de servico (vLLM, TGI)
- [ ] API publica de inferencia
- [ ] Paper publicado em conferencia revisada por pares
- [ ] Geodesic attention otimizada (CUDA kernels customizados)

---

Este roadmap e indicativo e pode mudar conforme prioridades e recursos.
Sugestoes sao bem-vindas via issues.
