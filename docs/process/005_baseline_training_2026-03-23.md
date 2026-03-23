# 005 - Baseline de Treino Pequeno e Estavel

**Data:** 2026-03-23
**Projeto:** drm_transformer
**Status:** concluido

## Objetivo

Definir baseline canonico "small" (1M params) com dataset fixo e publico,
pipeline de treino estavel com early stop, validacao periodica, e entregaveis
automaticos (checkpoint, metrics.json, training_log.json).

## Alteracoes

- `configs/baselines/small_1m.yaml` (novo) - Config baseline canonico
- `scripts/prepare_baseline_data.py` (novo) - Gera dataset fixo Wikipedia EN com SHA256
- `src/drm_transformer/training/trainer.py` - Early stop, val metrics no log, metrics.json final
- `scripts/train_distributed.py` - eval_data_dir da config YAML

## Decisoes Tecnicas

- **Wikipedia EN como baseline**: publico, sem auth, reproduzivel por qualquer pessoa
- **10M tokens**: pequeno o suficiente para rodar rapido, grande o suficiente para convergir
- **Train/val split 90/10**: deterministico (primeiros 90% train, ultimos 10% val)
- **SHA256 por shard**: verificacao de integridade via `--verify`
- **Early stop patience=10**: para se val_loss nao melhora em 10 avaliacoes consecutivas
- **metrics.json**: sumario final com best_val_loss, best_val_ppl, total_steps, tokens/s

## Proximos Passos

- Rodar baseline completo e verificar reprodutibilidade
- Comparar 2 runs com --deterministic para validar criterio de sucesso
