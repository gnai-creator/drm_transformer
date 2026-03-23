# 006 - Avaliacao Padronizada e Ablacoes

**Data:** 2026-03-23
**Projeto:** drm_transformer
**Status:** parcial (infra pronta, execucao depende de dataset)

## Objetivo

Criar suite de avaliacao automatizada e matriz de ablacoes para medir
contribuicao de cada componente DRM (gravity, gamma, variable_dim).

## Alteracoes

- `configs/ablations/` (novo) - 4 configs: full, no_gravity, no_gamma, no_variable_dim
- `scripts/run_ablations.py` (novo) - Roda todas as ablacoes e gera results_ablations.md
- `scripts/eval_standard.py` (novo) - Avaliacao padronizada (perplexity, futuro: HellaSwag/ARC)
- `docs/future/001_commonsense_benchmarks_2026-03-23.md` (novo) - Plan para benchmarks em escala

## Decisoes Tecnicas

- **Ablacoes no baseline 1M**: rapido de rodar, valida contribuicao de cada componente
- **Perplexity como metrica primaria**: nao requer escala, funciona com qualquer tamanho
- **HellaSwag/ARC deferidos**: precisam de modelo 350M+ para resultados significativos
- **results_ablations.md** gerado automaticamente: reproduzivel via um comando

## Proximos Passos

- Rodar ablacoes apos dataset baseline pronto
- Implementar HellaSwag/ARC apos treino do 350M
- Comparar com GPT-2 124M como referencia publica
