# 004 - Reprodutibilidade e Higiene de Experimento

**Data:** 2026-03-23
**Projeto:** drm_transformer
**Status:** concluido

## Objetivo

Implementar infraestrutura de reprodutibilidade minima: seed global, determinismo,
run manifest automatico, lock de dependencias e guia de reproducao.

## Alteracoes

- `src/drm_transformer/training/reproducibility.py` (novo) - set_seed, set_deterministic, build_run_manifest
- `scripts/train_distributed.py` - --seed e --deterministic args, manifest automatico
- `requirements-lock.txt` (novo) - lock completo de dependencias
- `repro.md` (novo) - guia passo-a-passo de reproducao

## Decisoes Tecnicas

- **warn_only=True** no `use_deterministic_algorithms`: evita crash em ops sem implementacao deterministica, apenas emite warning
- **CUBLAS_WORKSPACE_CONFIG=:4096:8**: necessario para determinismo em matmul CUDA
- **Manifest JSON por run**: captura git hash, config hash, hardware, deps — permite rastrear qualquer experimento
- **requirements-lock.txt** em vez de poetry/uv: simplicidade, sem tooling adicional

## Proximos Passos

- Validar determinismo end-to-end com 2 runs identicos no 1m config
- Adicionar seed broadcast entre ranks no DDP
