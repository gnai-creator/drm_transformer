# DRM Transformer

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Commercial License](https://img.shields.io/badge/License-Commercial-orange.svg)](LICENSE-COMMERCIAL.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Configs](https://img.shields.io/badge/Scaling-1M%20to%20640B-green.svg)](configs/scaling/)
[![Architecture](https://img.shields.io/badge/Attention-Geodesic-blueviolet.svg)](#inovacoes-principais)
[![Papers](https://img.shields.io/badge/Papers-3%20DRM-yellow.svg)](#papers)

Decoder-only Transformer onde o espaco de embeddings vive num Directional
Relational Manifold (DRM) com tensor metrico aprendido G(x) dependente de
posicao, curvatura gravitacional derivada da massa dos tokens, e
dimensionalidade variavel por token. A atencao padrao (dot-product) e
substituida por Geodesic Attention: a distancia entre queries e keys e
computada sob G(x), e o fator de escala segue dinamica relativistica
(gamma-scaling do fator de Lorentz).

---

## Inovacoes Principais

1. **Geodesic Attention** - Distancia geodesica sob G(x) substitui dot-product
   Euclidiano. Tokens proximos no manifold recebem mais atencao.
2. **MetricNet** - Tensor metrico G(x) aprendido via MLP + Cholesky (SPD
   garantido). A geometria do espaco de embeddings e aprendida end-to-end.
3. **Gravitational Token Embedding** - Cada token possui massa aprendida que
   deforma G(x) na vizinhanca, atraindo tokens vizinhos (analogia com gravidade).
4. **DimensionalGate** - Dimensionalidade efetiva dimD(p) varia por token via
   mascara suave. Tokens simples usam poucas dimensoes, tokens complexos usam mais.
5. **Gamma-Scaling (Relativistic Dynamics)** - Fator de Lorentz gamma escala
   adaptativamente a resolucao metrica conforme a distancia aos anchors no manifold.

---

## Arquitetura

```txt
 input_ids [B, T]
      |
      v
 +------------------+
 |  Token Embedding  |
 +------------------+
      |
      v
 +------------------+
 | DimensionalGate  |  --> dimD(p) por token (mascara suave)
 +------------------+
      |
      v
 +==========================================+
 |  DRM Block x N (Pre-RMSNorm + Residual)  |
 |                                          |
 |  1. Projetar Q, K para manifold [0,1]^d  |
 |     q_m = sigmoid(W_q @ q_head)          |
 |     k_m = sigmoid(W_k @ k_head)          |
 |                                          |
 |  2. G(x) = MetricNet(q_m)  [SPD, Chol.]  |
 |     G_grav = GravityField(G, mass, q_m)  |
 |                                          |
 |  3. d^2 = (q_m - k_m)^T G_grav (q_m-k_m) |
 |     d^2 *= gamma^2 (se gamma_enabled)    |
 |                                          |
 |  4. attn = softmax(-d^2 / temp) @ V      |
 |                                          |
 |  5. FFN (SwiGLU) + Residual              |
 +==========================================+
      |
      v
 +------------------+
 |   RMSNorm final  |
 +------------------+
      |
      v
 +------------------+
 |     LM Head      |  (weight-tied com Token Embedding)
 +------------------+
      |
      v
  logits [B, T, V]
```

---

## Quick Start

### Instalacao

```bash
git clone https://github.com/gnai-creator/drm_transformer.git
cd drm_transformer
pip install -e ".[dev,data,eval]"
```

### Preparar Dados

```bash
# Monolingual (English Wikipedia, GPT-2 tokenizer)
python scripts/prepare_multilingual_data.py \
    --output-dir data/en \
    --max-tokens 10000000 \
    --vocab-size 50257 \
    --langs en

# Multilingual (Wikipedia 5 linguas, o200k_base subset 50K)
python scripts/prepare_multilingual_data.py \
    --output-dir data/multilingual \
    --max-tokens 50000000 \
    --vocab-size 50000 \
    --langs en,pt,es,fr,de
```

### Treinamento

```bash
# Single GPU
python scripts/train_distributed.py \
    --config configs/scaling/15m.yaml \
    --data-dir data/

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 scripts/train_distributed.py \
    --config configs/scaling/350m.yaml \
    --data-dir data/

# FSDP (13B+)
torchrun --nproc_per_node=8 scripts/train_distributed.py \
    --config configs/scaling/13b.yaml \
    --data-dir data/

# Resume
python scripts/train_distributed.py \
    --config configs/scaling/350m.yaml \
    --resume auto
```

### Geracao

```python
import torch
from drm_transformer import DRMTransformerConfig, DRMTransformer

config = DRMTransformerConfig(d_model=256, n_layers=4, n_heads=4, d_manifold=8)
model = DRMTransformer(config)

prompt = torch.randint(0, config.vocab_size, (1, 10))
generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=50)
print(generated.shape)  # [1, 60]
```

### Forward pass

```python
input_ids = torch.randint(0, config.vocab_size, (2, 128))
targets = torch.randint(0, config.vocab_size, (2, 128))

logits, loss = model(input_ids, targets)
# logits: [2, 128, 50257]
# loss: escalar (cross-entropy)
```

### Voronoi Foliation (Avaliacao Topologica)

```bash
# 1. Extrair vectores DRM
python scripts/extract_drm_vectors.py \
    --checkpoint checkpoints/1m/final.pt \
    --data-dir data/ \
    --output-dir eval_results/foliation_1m \
    --max-tokens 100000

# 2. Voronoi Foliation (9 fases: LTSA, Homology, Reeb, ARI, ...)
python scripts/voronoi_foliation_drm.py \
    --coords eval_results/foliation_1m/drm_coords.npy \
    --G-diag eval_results/foliation_1m/drm_G_diag.npy \
    --gamma eval_results/foliation_1m/drm_gamma.npy \
    --output-dir eval_results/foliation_1m \
    --n-seeds 30 \
    --homology-points 1500
```

Saidas: `foliation_results.json` com F-score, H1/H2, ARI, coherence, Reeb graph.
H1 alto num modelo sem treino indica topologia nao-trivial; F > 0.5 = foliation robusta.

### Avaliacao via API Python

```python
from drm_transformer.evaluation import DRMFoliationEvaluator
from drm_transformer.training.data import create_dataloader

evaluator = DRMFoliationEvaluator(model, device="cuda")
loader = create_dataloader("data/", seq_len=1024, batch_size=4)
results = evaluator.evaluate(loader, max_tokens=100_000)
print(f"F={results['foliation_score']:.4f}, topology={results['topology']}")
```

---

## Scaling Configs

| Config | Params | d_model | Layers | Heads | d_manifold | Context |
|--------|--------|---------|--------|-------|------------|---------|
| [1m](configs/scaling/1m.yaml) | ~1M | 64 | 4 | 2 | 4 | 256 |
| [15m](configs/scaling/15m.yaml) | ~15M | 256 | 6 | 4 | 8 | 512 |
| [50m](configs/scaling/50m.yaml) | ~50M | 512 | 8 | 8 | 12 | 1024 |
| [350m](configs/scaling/350m.yaml) | ~350M | 1024 | 24 | 16 | 16 | 1024 |
| [1.3b](configs/scaling/1.3b.yaml) | ~1.3B | 2048 | 24 | 16 | 20 | 2048 |
| [13b](configs/scaling/13b.yaml) | ~13B | 5120 | 40 | 40 | 24 | 4096 |
| [70b](configs/scaling/70b.yaml) | ~70B | 8192 | 80 | 64 | 28 | 4096 |
| [162b](configs/scaling/162b.yaml) | ~162B | 12288 | 96 | 96 | 32 | 4096 |
| [640b](configs/scaling/640b.yaml) | ~640B | 16384 | 126 | 128 | 40 | 8192 |

---

## Papers

Baseado em tres papers de Felipe Maya Muniz:

1. **DRM: Directional Relational Manifolds** - Geometria com dimensionalidade
   variavel, metrica relacional, convergencia toroidal.
   [DOI: 10.5281/zenodo.19058837](https://doi.org/10.5281/zenodo.19058837)

2. **The Geometry of Consciousness** - Dimensionalidade domina performance
   cognitiva (r=0.920). Ceiling theorem: O(S) <= d. Interacao d x CC.
   [DOI: 10.5281/zenodo.19059445](https://doi.org/10.5281/zenodo.19059445)

3. **DRM Relativistic Dynamics** - Fator de Lorentz como parametro de
   controle, bifurcation cascade, convergencia toroidal sob recorrencia.
   [DOI: 10.5281/zenodo.19140125](https://doi.org/10.5281/zenodo.19140125)

---

## Estrutura do Projeto

```txt
drm_transformer/
|-- README.md
|-- ARCHITECTURE.md
|-- LICENSE                    # AGPL-3.0
|-- LICENSE-COMMERCIAL.md      # Licenca comercial
|-- CLA.md                     # Contributor License Agreement
|-- PRIOR_ART.md               # Arte anterior
|-- ROADMAP.md
|-- SECURITY.md
|-- COPYRIGHT
|-- CITATION.cff
|-- pyproject.toml
|-- .gitignore
|
|-- src/drm_transformer/
|   |-- __init__.py            # Exports: DRMTransformerConfig, DRMTransformer
|   |-- config.py              # DRMTransformerConfig (dataclass)
|   |-- model.py               # DRMTransformer (modelo principal)
|   |-- attention.py           # DRMAttention + RotaryEmbedding + apply_rope
|   |-- metric_net.py          # MetricNet: G(x) via MLP + Cholesky (SPD)
|   |-- manifold.py            # ManifoldProjection + gamma_scale
|   |-- gravity.py             # GravityField: massa + deformacao metrica
|   |-- dimensional_gate.py    # DimensionalGate: dimD(p) variavel
|   |-- layers.py              # RMSNorm, FeedForward (SwiGLU), DRMTransformerBlock
|   |-- losses.py              # metric_regularization + metric_diversity_loss
|   |
|   +-- training/
|       |-- __init__.py
|       |-- distributed.py     # DDP, FSDP, mixed precision, grad checkpoint
|       |-- trainer.py         # DRMTrainer: loop de treino completo
|       +-- data.py            # ShardedDataset + create_dataloader
|
|   +-- evaluation/
|       |-- __init__.py
|       +-- foliation.py        # DRMFoliationEvaluator (pipeline completo)
|
|-- scripts/
|   |-- train_distributed.py    # Script de lancamento (single/multi GPU)
|   |-- extract_drm_vectors.py  # Extrai coords, G_diag, gamma, mass
|   +-- voronoi_foliation_drm.py # 9 fases: Voronoi, LTSA, Homology, Reeb, ARI
|
|-- configs/scaling/           # 9 configs: 1M, 15M, 50M, 350M, 1.3B, 13B, 70B, 162B, 640B
+-- docs/                      # Documentacao
```

---

## Licenca

Este projeto usa **licenciamento dual**:

### AGPL-3.0 (open-source)

O codigo esta licenciado sob [GNU Affero General Public License v3.0](LICENSE).
Voce pode usar, modificar e redistribuir livremente, desde que cumpra os termos
da AGPL (incluindo disponibilizar codigo-fonte de versoes modificadas usadas
em rede).

### Licenca Comercial

Para uso em projetos proprietarios, SaaS sem obrigacao de abrir codigo, ou
redistribuicao sem copyleft, uma [licenca comercial](LICENSE-COMMERCIAL.md)
esta disponivel.

Contato: felipe@truthagi.ai

### Contribuicoes

Contribuicoes externas requerem assinatura do [CLA](CLA.md) para manter a
viabilidade do licenciamento dual.

---

## Citacao

```bibtex
@software{muniz2026drm_transformer,
  author = {Muniz, Felipe Maya},
  title = {DRM Transformer: Decoder-only Transformer with Directional Relational Manifold Geometry},
  year = {2026},
  url = {https://github.com/gnai-creator/drm_transformer},
  license = {AGPL-3.0-or-later}
}
```

---

Copyright (C) 2026 Felipe Maya Muniz. All rights reserved.
