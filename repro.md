# Reprodutibilidade

Guia curto para reproduzir o baseline DRM Transformer 3.5M, as ablacoes e a
analise de foliacao Voronoi.

## 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

No PowerShell, ative com:

```powershell
.\.venv\Scripts\Activate.ps1
```

Para CUDA, instale PyTorch com wheel CUDA antes de rodar os treinos. Verifique:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 2. Experimento Completo

Um comando prepara/verifica os dados publicos, treina o baseline `full`, treina
as ablacoes, extrai vetores DRM e roda Voronoi em cada variante:

```bash
python scripts/run_ablation_foliation.py --prepare-data --prepare-max-tokens 10000000 --seed 42 --deterministic --device cuda --n-seeds 80 --homology-points 1200 --homology-long-bar-ratio 0.75 --homology-restarts 5
```

O dataset vem da Wikipedia publica e nao exige login no Hugging Face.

O `full` usa `configs/baselines/small_3.5M.yaml` e salva em
`checkpoints/baseline_3.5m/`. As outras variantes usam `configs/ablations/` e
salvam em `checkpoints/ablations/<ablacao>/`.

Se nao houver CUDA, use `--device cpu` ou remova o argumento para `auto`.

O script pula etapas ja prontas por padrao. Para continuar uma rodada longa:

```bash
python scripts/run_ablation_foliation.py --skip-train --device cuda
```

Para rodar so algumas variantes:

```bash
python scripts/run_ablation_foliation.py --only full,annealed_torus,generic_geometry --device cuda
```

Resumo final:

- `eval-results/ablations_foliation/summary.md`
- `eval-results/ablations_foliation/summary.json`
- `eval-results/ablations_foliation/<ablacao>/foliation_results.json`

Saidas principais do baseline:

- `checkpoints/baseline_3.5m/final.pt`
- `checkpoints/baseline_3.5m/best.pt`
- `checkpoints/baseline_3.5m/run_manifest.json`
- `checkpoints/baseline_3.5m/metrics.json`

### Comandos opcionais

Treinar apenas o baseline, sem ablações:

```bash
python scripts/train_distributed.py \
  --config configs/baselines/small_3.5M.yaml \
  --seed 42 \
  --deterministic \
  --device cuda
```

Preparar dados manualmente:

```bash
python scripts/prepare_baseline_data.py
python scripts/prepare_baseline_data.py --max-tokens 20000000
python scripts/prepare_baseline_data.py --verify
```

## 3. Foliacao Manual

Para avaliar apenas o baseline ja treinado:

```bash
python scripts/extract_drm_vectors.py \
  --checkpoint checkpoints/baseline_3.5m/final.pt \
  --data-dir data \
  --output-dir eval-results/foliation_3.5m \
  --max-tokens 100000 \
  --device cuda

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

## 4. Criterio Atual

A topologia toroidal so e considerada validada pelo criterio estrito quando a
homologia persistente retorna:

- `H1=2`
- `H2=1`
- `T2 stable >= 0.60`

Resultado atual do baseline 3.5M:

```text
seed-robust near-toroidal signature
random_init: Near T2=0.0
seed_42: H1=2, H2=1, T2 stable=0.4, Near T2=0.7
seed_123: H1=1, H2=1, T2 stable=0.2, Near T2=0.6
seed_2025: H1=2, H2=1, T2 stable=0.3, Near T2=0.8
```

Como nenhuma seed atingiu `T2 stable >= 0.60`, o resultado deve ser relatado
como assinatura toroidal quase-estavel, nao como toro estavel fechado.

## 5. Ablacoes de Treino

Para gerar apenas a tabela de metricas de treino/perplexidade:

```bash
python scripts/run_ablations.py --seed 42 --deterministic
python scripts/eval_standard.py --all-ablations
```

Resultados:

- `checkpoints/baseline_3.5m/results_ablations.md`
- `checkpoints/baseline_3.5m/results_ablations.json`

## 6. Controles Topologicos

Para consolidar random init, baseline, ablações e checkpoints intermediarios em
uma tabela unica:

```bash
python scripts/topology_controls.py --device cuda --force-foliation
```

Saidas:

- `eval-results/topology_controls/topology_controls.md`
- `eval-results/topology_controls/topology_controls.json`

Para treinar e comparar seeds adicionais:

```bash
python scripts/topology_controls.py --train-seeds 42,123,2025 --device cuda --deterministic
```

Para treinar completo:
```bash
python scripts/topology_controls.py --train-seeds 42,123,2025 --device cuda --deterministic --force-foliation
```

## 7. Notas

- Use a mesma seed, config e hardware para comparar runs.
- Multi-GPU pode introduzir diferencas pequenas por reducoes numericas.
- `--deterministic` melhora reprodutibilidade, mas pode reduzir performance.
