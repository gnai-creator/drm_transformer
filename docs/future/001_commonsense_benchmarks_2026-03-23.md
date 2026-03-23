# 001 - Benchmarks de Commonsense e QA

**Data:** 2026-03-23
**Projeto:** drm_transformer
**Status:** pendente (depende de modelo treinado em escala)

## Objetivo

Integrar benchmarks padrao de commonsense e QA ao pipeline de avaliacao
para comparacao com modelos de referencia (GPT-2, LLaMA, etc.).

## Benchmarks Planejados

| Benchmark | Tipo | Formato | Metrica |
|-----------|------|---------|---------|
| HellaSwag | Commonsense | Multiple choice (4 opcoes) | Accuracy |
| ARC-Easy | QA/Science | Multiple choice (4 opcoes) | Accuracy |
| ARC-Challenge | QA/Science | Multiple choice (4 opcoes) | Accuracy |
| MMLU | Knowledge | Multiple choice (4 opcoes) | Accuracy |

## Abordagem

Avaliacao via log-likelihood ranking (zero-shot):
1. Para cada pergunta, computar log-likelihood de cada opcao de resposta
2. Selecionar opcao com maior likelihood
3. Comparar com ground truth

Isso nao requer fine-tuning — funciona com o modelo pre-treinado.

## Pre-requisitos

- Modelo treinado com pelo menos 350M params e 1B+ tokens
  (modelos menores nao tem capacidade suficiente para benchmarks de commonsense)
- Dataset de treino completo (CulturaX 20B tokens em andamento)
- Implementar avaliacao log-likelihood em `scripts/eval_standard.py`

## Implementacao

Adicionar ao `scripts/eval_standard.py`:
- `eval_hellaswag()`: download do dataset via HF, avaliacao zero-shot
- `eval_arc()`: download ARC-Easy e ARC-Challenge via HF
- Flags: `--hellaswag`, `--arc`, `--mmlu`

Alternativa: usar framework lm-evaluation-harness (EleutherAI) que ja
tem todos esses benchmarks implementados:
```bash
pip install lm-eval
lm_eval --model hf --model_args pretrained=checkpoints/350m/final.pt --tasks hellaswag,arc_easy
```

## Quando Rodar

- Apos treino completo do 350M (primeiro modelo com escala suficiente)
- Incluir na tabela de ablacoes comparando full vs variantes
- Comparar com GPT-2 124M (referencia publica mais proxima em escala)
