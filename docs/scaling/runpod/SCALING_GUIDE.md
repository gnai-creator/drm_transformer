# DRM Transformer -- RunPod Scaling Guide

Guia completo de custo, tempo e estrategia de treinamento do DRM Transformer
no RunPod, de 1M a 640B params.

> **Nota**: precos RunPod sao dinamicos e carregados via JS. Os valores abaixo
> sao estimativas baseadas em precos publicos de referencia. Verificar sempre
> em [runpod.io/pricing](https://www.runpod.io/pricing) antes de alugar.
> Data de referencia: Marco 2026.

---

## 1. Precos RunPod por GPU

### GPUs Single

| GPU | VRAM | On-Demand ($/hr) | Community ($/hr) | Notas |
|-----|------|------------------|------------------|-------|
| RTX 4090 | 24 GB | ~0.44 | ~0.29 | Consumer, PCIe |
| RTX 6000 Ada | 48 GB | ~0.68 | ~0.44 | Professional |
| L40S | 48 GB | ~0.74 | ~0.49 | Data center, ada arch |
| L40 | 48 GB | ~0.59 | ~0.39 | Data center |
| L4 | 24 GB | ~0.32 | ~0.21 | Inference-focused |
| A100 PCIe 40 GB | 40 GB | ~1.04 | ~0.69 | Legacy |
| A100 PCIe 80 GB | 80 GB | ~1.44 | ~0.89 | Popular para treino |
| A100 SXM4 80 GB | 80 GB | ~1.64 | ~1.09 | NVLink para multi-GPU |
| H100 PCIe 80 GB | 80 GB | ~2.49 | ~1.89 | Gen5 PCIe |
| H100 SXM5 80 GB | 80 GB | ~3.29 | ~2.49 | NVLink 600GB/s |
| H200 SXM 141 GB | 141 GB | ~4.49 | ~3.49 | HBM3e, best single-GPU |

### Pods Multi-GPU

| Config | On-Demand ($/hr) | Community ($/hr) | Interconnect |
|--------|------------------|------------------|--------------|
| 2x A100 80GB SXM | ~3.28 | ~2.18 | NVLink |
| 4x A100 80GB SXM | ~6.56 | ~4.36 | NVLink |
| 8x A100 80GB SXM | ~13.12 | ~8.72 | NVLink |
| 2x H100 SXM5 | ~6.58 | ~4.98 | NVLink 600GB/s |
| 4x H100 SXM5 | ~13.16 | ~9.96 | NVLink 600GB/s |
| 8x H100 SXM5 | ~26.32 | ~19.92 | NVLink 600GB/s |
| 2x H200 SXM | ~8.98 | ~6.98 | NVLink |
| 4x H200 SXM | ~17.96 | ~13.96 | NVLink |
| 8x H200 SXM | ~35.92 | ~27.92 | NVLink |

### Armazenamento

| Tipo | Custo |
|------|-------|
| Volume persistente (pod ativo) | $0.10/GB/mes |
| Volume persistente (pod parado) | $0.20/GB/mes |
| Network Storage (< 1TB) | $0.07/GB/mes |
| Network Storage (> 1TB) | $0.05/GB/mes |

---

## 2. Requisitos de Memoria por Modelo

| Config | Params | bf16 (GB) | Treino bf16 (GB) | Treino + GradCkpt (GB) | GPU Minima |
|--------|--------|-----------|------------------|------------------------|------------|
| 1M | ~1M | 0.01 | 0.05 | 0.03 | RTX 4090 |
| 5M | ~5.7M | 0.01 | 0.08 | 0.04 | RTX 4090 |
| 10M | ~9.7M | 0.02 | 0.13 | 0.07 | RTX 4090 |
| 15M | ~19M | 0.04 | 0.27 | 0.14 | RTX 4090 |
| 50M | ~50M | 0.10 | 0.70 | 0.36 | RTX 4090 |
| 350M | ~350M | 0.70 | 4.9 | 2.5 | RTX 4090 |
| 1.3B | ~1.3B | 2.6 | 18.2 | 9.4 | RTX 4090 |
| 13B | ~13B | 26 | 182 | 94 | 2x H100 SXM / 2x H200 |
| 70B | ~70B | 140 | 980 | 504 | 8x H100 SXM / 4x H200 |
| 162B | ~162B | 324 | 2268 | 1166 | 8x H200 + FSDP |
| 640B | ~640B | 1280 | 8960 | 4608 | Multi-node |

**Formula de memoria:**
- bf16: params x 2 bytes
- Treino (bf16 + AdamW): params x 14 bytes (2 weights + 4 gradients + 8 optimizer)
- Com gradient checkpointing: params x 6 bytes x 1.2 (buffer activations)

**Nota DRM:** MetricNet, GravityField (RFF) e DimensionalGate adicionam ~2-5% de
memoria extra vs transformer padrao equivalente. O overhead principal e no compute, nao na memoria.

---

## 3. Throughput Estimado (tok/s)

### Fatores de correcao DRM

O DRM Transformer tem overhead sobre um transformer padrao equivalente:
- GeodesicAttention: ~2-3x mais matmuls que dot-product attention
- MetricNet: MLP + Cholesky por token por layer
- GravityField (RFF): O(T x R) por layer (R=64, leve)
- DimensionalGate: gate sigmoid (leve)

**Fator DRM por escala:**
- 1M-50M: 0.55x (overhead relativo alto, modelo pequeno)
- 350M-1.3B: 0.50x
- 13B+: 0.45x (mais layers = mais overhead cumulativo)

**Fator por seq_len (custo quadratico da geodesic attention):**
- 256: 1.0x | 512: 0.85x | 1024: 0.65x | 2048: 0.40x | 4096: 0.20x | 8192: 0.10x

### Matriz de Throughput (tok/s estimado)

| Config | RTX 4090 | A100 40G | A100 80G | H100 PCIe | H100 SXM | H200 |
|--------|----------|----------|----------|-----------|----------|------|
| 1M | 44K | 66K | 77K | 110K | 154K | 193K |
| 5M | 37K | 56K | 65K | 94K | 131K | 163K |
| 10M | 32K | 48K | 56K | 80K | 112K | 140K |
| 15M | 26K | 40K | 46K | 66K | 92K | 116K |
| 50M | 17K | 26K | 30K | 43K | 60K | 75K |
| 350M | 14K | 21K | 24K | 34K | 48K | 60K |
| 1.3B | OOM* | 8K | 12K | 18K | 24K | 32K |
| 13B | OOM | OOM | OOM | OOM | 6K** | 8K** |
| 70B | OOM | OOM | OOM | OOM | OOM | OOM |

\* 1.3B cabe na RTX 4090 com batch=1 + grad_ckpt, mas throughput muito baixo (~4K)
\** Requer 2+ GPUs com FSDP

### Multi-GPU Throughput (FSDP/DDP)

Speedup esperado com NVLink:
- 2x: 1.9x | 4x: 3.7x | 8x: 7.2x

Com PCIe:
- 2x: 1.7x | 4x: 3.0x | 8x: 5.5x

| Config | 2x H100 SXM | 4x H100 SXM | 8x H100 SXM | 4x H200 | 8x H200 |
|--------|-------------|-------------|-------------|---------|---------|
| 13B | 11K | 22K | 43K | 30K | 58K |
| 70B | OOM | 8K | 18K | 12K | 24K |
| 162B | OOM | OOM | 6K | OOM | 10K |
| 640B | OOM | OOM | OOM | OOM | OOM* |

\* 640B requer multi-node (16+ H200) ou MoE

---

## 4. Tokens Necessarios por Modelo

Chinchilla scaling x 1.3 safety factor (arquitetura nova):

| Config | Params | Tokens (validacao) | Tokens (producao) | Dados (GB comprimido) |
|--------|--------|-------------------|-------------------|-----------------------|
| 1M | 1M | 20M | 50M | 0.04 |
| 5M | 5.7M | 100M | 150M | 0.3 |
| 10M | 9.7M | 200M | 300M | 0.5 |
| 15M | 19M | 300M | 500M | 1.0 |
| 50M | 50M | 1B | 2B | 4 |
| 350M | 350M | 7B | 9B | 18 |
| 1.3B | 1.3B | 26B | 34B | 68 |
| 13B | 13B | 260B | 340B | 680 |
| 70B | 70B | 1.4T | 1.8T | 3600 |
| 162B | 162B | 3.2T | 4.2T | 8400 |
| 640B | 640B | 12.8T | 16.6T | 33200 |

---

## 5. Estrategias por Budget Mensal

### 5.1 Budget: $100/mes

**GPU Recomendada:** 1x RTX 4090 (Community)
**Custo/hora:** ~$0.29
**Horas disponiveis/mes:** ~345h

| Modelo | Tok/s | Tokens/mes | Chinchilla | Meses (validacao) | Meses (producao) | Custo total |
|--------|-------|-----------|------------|-------------------|-------------------|-------------|
| 1M | 44K | 55B | 20M | < 1 dia | < 1 dia | $1 |
| 5M | 37K | 46B | 100M | < 1 dia | < 1 dia | $2 |
| 10M | 32K | 40B | 200M | < 1 dia | < 1 dia | $3 |
| 15M | 26K | 32B | 300M | < 1 dia | < 1 dia | $5 |
| 50M | 17K | 21B | 1B | 1 dia | 2 dias | $15 |
| 350M | 14K | 17B | 7B | 6 dias | 8 dias | $50 |

**Estrategia recomendada:**
- Treinar 1M, 15M e 50M para validacao topologica (H1, ARI, foliation score)
- 350M viavel para validacao (7B tokens em ~6 dias)
- Usar Community Cloud (Spot) para 40% desconto
- `save_interval: 100` para tolerancia a interrupcoes Spot

```bash
python scripts/train_distributed.py \
    --config configs/scaling/multilingual/50m.yaml \
    --data-dir /workspace/data/multilingual
```

### 5.2 Budget: $200/mes

**GPU Recomendada:** 1x A100 80GB (Community)
**Custo/hora:** ~$0.89
**Horas disponiveis/mes:** ~225h

| Modelo | Tok/s | Tokens/mes | Chinchilla | Meses (validacao) | Meses (producao) |
|--------|-------|-----------|------------|-------------------|-------------------|
| 50M | 30K | 24B | 1B | < 1 dia | 2 dias |
| 350M | 24K | 19B | 7B | 4 dias | 5 dias |
| 1.3B | 12K | 10B | 26B | 3 meses | 4 meses |

**Estrategia:** Focar no 350M. 1.3B viavel mas lento (3-4 meses para validacao).

### 5.3 Budget: $300/mes

**GPU Recomendada:** 1x H100 PCIe (Community)
**Custo/hora:** ~$1.89
**Horas disponiveis/mes:** ~159h

| Modelo | Tok/s | Tokens/mes | Chinchilla | Meses (validacao) | Meses (producao) |
|--------|-------|-----------|------------|-------------------|-------------------|
| 350M | 34K | 19B | 7B | 3 dias | 4 dias |
| 1.3B | 18K | 10B | 26B | 3 meses | 4 meses |

**Estrategia:** 350M em producao + iniciar validacao do 1.3B.

### 5.4 Budget: $500/mes

**GPU Recomendada:** 1x H100 SXM5 (Community) ou 1x H200
**Custo/hora:** ~$2.49 (H100 SXM) ou ~$3.49 (H200)
**Horas disponiveis/mes:** ~200h (H100) ou ~143h (H200)

| Modelo | GPU | Tok/s | Tokens/mes | Meses (validacao) | Meses (producao) |
|--------|-----|-------|-----------|-------------------|-------------------|
| 350M | H100 SXM | 48K | 35B | 2 dias | 3 dias |
| 1.3B | H100 SXM | 24K | 17B | 2 meses | 2 meses |
| 1.3B | H200 | 32K | 16B | 2 meses | 3 meses |

**Estrategia:** H100 SXM e melhor custo-beneficio neste budget. 1.3B em 2 meses.

### 5.5 Budget: $1000/mes

**GPU Recomendada:** 2x H100 SXM5 (Community)
**Custo/hora:** ~$4.98
**Horas disponiveis/mes:** ~200h

| Modelo | Tok/s | Tokens/mes | Meses (validacao) | Meses (producao) |
|--------|-------|-----------|-------------------|-------------------|
| 1.3B | 46K | 33B | 1 mes | 1 mes |
| 13B | 11K | 8B | 33 meses | 43 meses |

**Estrategia:** 1.3B em producao em 1 mes. 13B nao viavel neste budget.

### 5.6 Budget: $1500/mes

**GPU Recomendada:** 4x H100 SXM5 (Community)
**Custo/hora:** ~$9.96
**Horas disponiveis/mes:** ~150h

| Modelo | Tok/s | Tokens/mes | Meses (validacao) | Meses (producao) |
|--------|-------|-----------|-------------------|-------------------|
| 1.3B | 89K | 48B | 2 semanas | 3 semanas |
| 13B | 22K | 12B | 22 meses | 28 meses |

**Estrategia:** 1.3B rapido. Para 13B, considerar spot + runs longos.

---

## 6. Estrategia Multi-GPU

### Quando usar multi-GPU

| Modelo | Single GPU | DDP (2+) | FSDP |
|--------|-----------|----------|------|
| <= 50M | Suficiente | Desnecessario | Desnecessario |
| 350M | Suficiente | Opcional (speedup) | Desnecessario |
| 1.3B | Possivel (H200) | Recomendado | Opcional |
| 13B | Impossivel | FSDP obrigatorio | Obrigatorio |
| 70B+ | Impossivel | Impossivel single-node | FSDP multi-node |

### Comandos

```bash
# DDP (2 GPUs, mesmo node)
torchrun --nproc_per_node=2 scripts/train_distributed.py \
    --config configs/scaling/multilingual/1.3b.yaml \
    --data-dir /workspace/data/multilingual

# FSDP (8 GPUs, 13B)
torchrun --nproc_per_node=8 scripts/train_distributed.py \
    --config configs/scaling/multilingual/13b.yaml \
    --data-dir /workspace/data/multilingual
```

### NVLink vs PCIe

Para FSDP (13B+), NVLink e critico:
- H100 SXM: NVLink 600 GB/s -> scaling quase linear
- H100 PCIe: PCIe Gen5 128 GB/s -> scaling 70% eficiente
- Break-even: 1x H200 ($3.49/hr) vs 2x H100 PCIe ($3.78/hr) -> H200 melhor para <= 1.3B

---

## 7. Cronograma de Scaling por Fase

### Fase 1: Validacao Topologica (meses 1-2)

**Objetivo:** Confirmar H1 convergencia no 15M e 50M
**Budget:** $100-200/mes
**GPU:** RTX 4090 ou A100 40GB (single)
**Metricas alvo:** H1 > 20, ARI > 0.7, foliation_score > 0.4

```bash
# Treinar 15M
python scripts/train_distributed.py \
    --config configs/scaling/multilingual/15m.yaml \
    --data-dir /workspace/data/multilingual

# Extrair + foliation
python scripts/extract_drm_vectors.py \
    --checkpoint checkpoints/multilingual_15m/final.pt \
    --data-dir /workspace/data/multilingual \
    --output-dir eval-results/foliation/multilingual_15m

python scripts/voronoi_foliation_drm.py \
    --coords eval-results/foliation/multilingual_15m/drm_coords.npy \
    --G-diag eval-results/foliation/multilingual_15m/drm_G_diag.npy \
    --gamma eval-results/foliation/multilingual_15m/drm_gamma.npy \
    --output-dir eval-results/foliation/multilingual_15m
```

### Fase 2: Scaling Bridge (meses 3-4)

**Objetivo:** 350M com topologia validada
**Budget:** $200-500/mes
**GPU:** A100 80GB ou H100 PCIe (single)

### Fase 3: Producao 1.3B (meses 5-8)

**Objetivo:** 1.3B com 26B+ tokens
**Budget:** $500-1000/mes
**GPU:** H100 SXM ou 2x A100 80GB

### Fase 4: Escala 13B+ (meses 9+)

**Objetivo:** 13B com topologia robusta
**Budget:** $1000-1500/mes
**GPU:** 4x H100 SXM ou 4x H200

---

## 8. Configuracao RunPod

### Template de Pod

```yaml
# RunPod Pod Config
container_image: runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04
volume_size: 100  # GB (dados + checkpoints)
gpu_type: NVIDIA H100 SXM5 80GB
gpu_count: 1
```

### Script de Setup

```bash
#!/bin/bash
# setup.sh - executar ao iniciar o pod

cd /workspace

# Clone repo
git clone https://github.com/gnai-creator/drm_transformer.git
cd drm_transformer
pip install -e ".[dev,data,eval]"

# Preparar dados (se nao existem no volume)
if [ ! -d "/workspace/data/multilingual" ]; then
    python scripts/prepare_multilingual_data.py \
        --output-dir /workspace/data/multilingual \
        --max-tokens 500000000 \
        --vocab-size 50000 \
        --langs en,pt,es,fr,de
fi

echo "[READY] DRM Transformer configurado"
```

### Variaveis de Ambiente

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0  # ou 0,1,2,3 para multi-GPU
export OMP_NUM_THREADS=4
```

### Volume Persistente

Guardar no volume:
- `/workspace/data/` - shards tokenizados (~1-50 GB)
- `/workspace/drm_transformer/checkpoints/` - checkpoints (~0.5-5 GB por save)
- `/workspace/drm_transformer/eval-results/` - resultados de foliation

Custo de storage para 100 GB: $10/mes (ativo) ou $20/mes (idle).

### Spot/Community: Tolerancia a Interrupcoes

```yaml
# Config ajustado para Spot (salvar frequente)
save_interval: 100     # a cada 100 steps (vs 500 default)
save_total_limit: 3    # manter so 3 checkpoints
log_interval: 25       # log mais frequente
```

```bash
# Resume automatico apos interrupcao
python scripts/train_distributed.py \
    --config configs/scaling/multilingual/350m.yaml \
    --data-dir /workspace/data/multilingual \
    --resume auto
```

---

## 9. Inferencia em Producao

### Latencia por Token (estimada)

| Modelo | RTX 4090 | A100 80G | H100 SXM | H200 |
|--------|----------|----------|----------|------|
| 350M | ~15 ms | ~8 ms | ~5 ms | ~4 ms |
| 1.3B | ~40 ms | ~20 ms | ~12 ms | ~9 ms |
| 13B | OOM | ~80 ms | ~45 ms | ~35 ms |

DRM adiciona ~50-80% de latencia vs transformer padrao na inferencia
(MetricNet + Cholesky + geodesic distance por token gerado).

### GPU Minima para Serve

| Modelo | bf16 | int8 | GPU Minima (bf16) | GPU Minima (int8) |
|--------|------|------|-------------------|-------------------|
| 350M | 0.7 GB | 0.35 GB | RTX 4090 | RTX 4090 |
| 1.3B | 2.6 GB | 1.3 GB | RTX 4090 | RTX 4090 |
| 13B | 26 GB | 13 GB | A100 40G | RTX 4090 |
| 70B | 140 GB | 70 GB | H200 | A100 80G |

### Quantizacao e Geometria

**Cuidado:** quantizacao int8/GPTQ pode degradar a precisao do MetricNet
(Cholesky requer boa precisao numerica para manter SPD). Recomendado:
- Quantizar transformer body (attention, FFN) normalmente
- Manter MetricNet e GravityField em bf16/fp16
- Testar H1 e foliation score apos quantizacao

---

## 10. Tabela Resumo Executiva

| Budget/mes | GPU Setup | Modelo Viavel | Tempo Validacao | Tempo Producao | Investimento Total |
|------------|-----------|---------------|-----------------|----------------|-------------------|
| $100 | 1x RTX 4090 (Spot) | 50M | 1 dia | 2 dias | $15 |
| $100 | 1x RTX 4090 (Spot) | 350M | 6 dias | 8 dias | $50 |
| $200 | 1x A100 80G (Spot) | 350M | 4 dias | 5 dias | $65 |
| $200 | 1x A100 80G (Spot) | 1.3B | 3 meses | 4 meses | $800 |
| $300 | 1x H100 PCIe (Spot) | 1.3B | 3 meses | 4 meses | $1200 |
| $500 | 1x H100 SXM (Spot) | 1.3B | 2 meses | 2 meses | $1000 |
| $1000 | 2x H100 SXM (Spot) | 1.3B | 1 mes | 1 mes | $1000 |
| $1000 | 2x H100 SXM (Spot) | 13B | 33 meses | 43 meses | $33K |
| $1500 | 4x H100 SXM (Spot) | 1.3B | 2 semanas | 3 semanas | $750 |
| $1500 | 4x H100 SXM (Spot) | 13B | 22 meses | 28 meses | $33K |

**Conclusao:** O sweet spot para validacao topologica e $100-200/mes com RTX 4090
ou A100, treinando ate 350M. Para producao 1.3B, $500-1000/mes com H100 SXM
por 1-2 meses. 13B+ requer investimento significativo (>$30K) ou acesso a
clusters HPC.

---

## Apendice: Links Uteis

- [RunPod Pricing](https://www.runpod.io/pricing) -- precos atualizados
- [RunPod GPU Cloud](https://www.runpod.io/gpu-cloud) -- disponibilidade
- [DRM Transformer](https://github.com/gnai-creator/drm_transformer) -- repo
- [RunPod Docs](https://docs.runpod.io/) -- documentacao
