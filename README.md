# DRM Transformer

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Commercial License](https://img.shields.io/badge/License-Commercial-orange.svg)](LICENSE-COMMERCIAL.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Configs](https://img.shields.io/badge/Scaling-1M%20to%20640B-green.svg)](configs/scaling/)
[![Architecture](https://img.shields.io/badge/Attention-Geodesic-blueviolet.svg)](#inovacoes-principais)
[![Papers](https://img.shields.io/badge/Papers-3%20DRM-yellow.svg)](#papers)

## Indice

- [Transformer Padrao vs DRM Transformer](#transformer-padrao-vs-drm-transformer)
- [Inovacoes Principais](#inovacoes-principais)
- [Arquitetura](#arquitetura)
- [DRMTransformerConfig](#drmtransformerconfig)
- [Loss Functions](#loss-functions-regularizacao-geometrica)
- [Quick Start](#quick-start)
- [Scaling Configs](#scaling-configs)
- [Papers](#papers)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Licenca](#licenca)
- [Citacao](#citacao)

---

Decoder-only Transformer onde o espaco de embeddings vive num Directional
Relational Manifold (DRM) com tensor metrico aprendido G(x) dependente de
posicao, curvatura gravitacional derivada da massa dos tokens, e
dimensionalidade variavel por token. A atencao padrao (dot-product) e
substituida por Geodesic Attention: a distancia entre queries e keys e
computada sob G(x), e o fator de escala segue dinamica relativistica
(gamma-scaling do fator de Lorentz).

---

## Transformer Padrao vs DRM Transformer

| Componente | Transformer Padrao | DRM Transformer | Analogia |
|---|---|---|---|
| **Espaco de embeddings** | R^d plano, Euclidiano | Manifold curvo com metrica G(x) | Mapa plano vs superficie da Terra: distancias reais dependem do terreno |
| **Attention score** | dot-product: q^T k | Distancia geodesica: -d_G(q,k)/temp | Medir "proximidade" em linha recta vs seguir o caminho real pelo terreno |
| **Metrica** | Fixa (identidade implicita) | Aprendida por posicao: G(x) = I + U(x) U(x)^T (low-rank) | Regua rigida vs regua elastica que estica conforme o lugar |
| **Tokens de alta informacao** | Tratados igual a todos os outros | Deformam a metrica local (gravidade) | Estrela que curva o espaco-tempo a sua volta, atraindo o que esta perto |
| **Dimensionalidade** | Fixa: todos os tokens usam d dimensoes | Variavel: dimD(p) por token via gate suave | Sala de d portas onde tokens simples abrem 3 e tokens complexos abrem todas |
| **Escala de resolucao** | Uniforme em todo o espaco | Gamma-scaling: regioes distantes dos anchors recebem mais resolucao | Microscopio que amplia automaticamente as zonas menos exploradas |
| **Posicao** | RoPE sobre embeddings | RoPE + coordenadas no manifold [0,1]^d via sigmoid | Endereco na rua (RoPE) + coordenadas GPS no mapa (manifold) |

**Em resumo:** um Transformer padrao opera num espaco plano e rigido.
O DRM Transformer opera num espaco que se curva, estica e adapta
conforme o que esta a processar -- como a diferenca entre navegar
num mapa plano e navegar na superficie real de um planeta.

---

## Inovacoes Principais

1. **Geodesic Attention** -- Distancia geodesica sob G(x) substitui dot-product
   Euclidiano. Tokens proximos no manifold recebem mais atencao.
2. **MetricNet** -- Tensor metrico G(x) = I + U(x)U(x)^T aprendido via MLP
   com factorizacao low-rank (rank=4). SPD garantido por construcao. Output
   zero-inicializado (G(x)=I no inicio, estabilidade). Geometria end-to-end.
3. **Gravitational Token Embedding** -- Cada token possui massa aprendida
   (via mass_net + Softplus) que deforma U(x) na vizinhanca via Random Fourier
   Features (RFF). Complexidade O(T*n_rff) em vez de O(T^2).
4. **DimensionalGate** -- Dimensionalidade efetiva dimD(p) varia por token via
   mascara suave. Tokens simples usam poucas dimensoes, tokens complexos usam mais.
5. **Gamma-Scaling (Relativistic Dynamics)** -- Fator de Lorentz gamma escala
   adaptativamente a resolucao metrica conforme a distancia aos anchors no manifold.
6. **Semantic Anchors** -- 6 pontos de referencia no manifold com significado
   interpretavel (truth, ignorance, safety, complexity, creativity, grounding).
   Inicializados com posicoes semanticas, aprendiveis pelo optimizer.

### Anchors Semanticos

| Anchor | Significado | Efeito no gamma-scaling |
|--------|------------|------------------------|
| truth | Baixa incerteza, alta qualidade | Tokens proximos: gamma ~1 (espaco normal) |
| ignorance | Alta incerteza epistemica | Tokens proximos: gamma ~1; longe: espaco expande |
| safety | Regiao de seguranca | Tokens longe de safety: gamma alto, mais resolucao |
| complexity | Alta complexidade de dominio | Referencia para complexidade estrutural |
| creativity | Exploracao, novidade | Zona de baixa restricao geometrica |
| grounding | Factualidade, evidencia | Ancora para tokens factuais |

Os anchors sao `nn.Parameter` -- o optimizer pode move-los durante o treino.
A inicializacao semantica fornece um prior geometrico interpretavel.

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
 |  2. U(x) = MetricNet(q_m) [d_manifold, r]|
 |     G(x) = I + U(x) U(x)^T  (SPD, lr)    |
 |     U_grav = GravityField.deform_U(U,q_m)|
 |                                          |
 |  3. delta = q_m - k_m                    |
 |     d^2 = ||delta||^2 + ||U^T delta||^2  |
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

## DRMTransformerConfig

| Campo | Tipo | Default | Descricao |
|-------|------|---------|-----------|
| `vocab_size` | int | 50257 | Tamanho do vocabulario |
| `max_seq_len` | int | 1024 | Comprimento maximo da sequencia |
| `d_model` | int | 384 | Dimensao dos embeddings |
| `n_layers` | int | 6 | Numero de blocos transformer |
| `n_heads` | int | 6 | Numero de heads de atencao |
| `d_ff` | int | 1536 | Dimensao hidden do FFN |
| `dropout` | float | 0.1 | Taxa de dropout |
| `bias` | bool | False | Usar bias nas camadas lineares |
| `d_manifold` | int | 16 | Dimensao do manifold |
| `metric_hidden` | int | 64 | Largura do hidden layer do MetricNet |
| `metric_rank` | int | 4 | Rank de U(x) em G(x) = I + UU^T |
| `n_quad` | int | 0 | Pontos de quadratura Gauss-Legendre (0 = Mahalanobis local) |
| `n_anchors` | int | 6 | Numero de anchors semanticos |
| `gamma_enabled` | bool | True | Habilitar gamma-scaling (Lorentz) |
| `gamma_c` | float | 4.0 | Limite de velocidade (c) |
| `gamma_alpha` | float | 0.0 | Alpha para annealing log-gamma |
| `temperature_init` | float | 1.0 | Temperatura inicial da atencao |
| `temperature_min` | float | 0.5 | Temperatura minima (clamp) |
| `gravity_enabled` | bool | True | Habilitar campo gravitacional |
| `gravity_strength` | float | 0.1 | Forca da gravidade sobre U(x) |
| `gravity_n_rff` | int | 64 | Random Fourier Features para gravidade |
| `variable_dim` | bool | True | Habilitar DimensionalGate por token |

---

## Loss Functions (Regularizacao Geometrica)

| Funcao | Descricao | Efeito |
|--------|-----------|--------|
| `metric_regularization(U)` | Penaliza norma excessiva de U(x) | Mantem G(x) proximo de I no inicio |
| `metric_diversity_loss(U)` | Encoraja variancia de U(x) entre tokens | Previne metrica estatica (position-independent) |
| `orthogonality_loss(U)` | Penaliza U^T U != I | Previne colapso dos eixos semanticos |
| `axis_variance_loss(U)` | Maximiza variancia de cada eixo na sequencia | Encoraja deformacao position-dependent |
| `anchor_alignment_loss(U, coords, anchors)` | Alinha primeiro eixo de U com anchor mais proximo | Sugere orientacao semantica (soft) |

Pesos configurados via `lambda_metric_reg`, `lambda_metric_diversity`, `lambda_ortho` no config de treino.

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
    --max-tokens 20000000000 \
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

### Validacao Empirica (testes + figuras)

```bash
# Pesos aleatorios (valida propriedades geometricas da arquitetura)
python empirical/tests/run_all.py

# Com checkpoint treinado (avalia modelo real)
python empirical/tests/run_all.py --checkpoint checkpoints/1m/final.pt

# Calibracao in-distribution (eval shard do mesmo dominio do treino)
python empirical/tests/run_all.py \
    --checkpoint checkpoints/multilingual_1m/run_1/final.pt \
    --eval-shard data/multilingual/shard_00001.npy \
    --output-dir eval_results/empirical_1m
```

Saidas:
- `empirical/results.json` — metricas de projecao, estatisticas dos eixos, separacao semantica, correlacao geometrica
- `empirical/figures/` — PCA, t-SNE, axis heatmap, axis separation, painel combinado 2x3

---

## Scaling Configs

| Config | Params | d_model | Layers | Heads | d_manifold | Context |
|--------|--------|---------|--------|-------|------------|---------|
| [1m](configs/scaling/1m.yaml) | ~1M | 64 | 4 | 2 | 4 | 256 |
| [5m](configs/scaling/5m.yaml) | ~5M | 96 | 6 | 3 | 4 | 512 |
| [10m](configs/scaling/10m.yaml) | ~10M | 160 | 4 | 4 | 6 | 512 |
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
|   |-- metric_net.py          # MetricNet: U(x) via MLP, G(x) = I + UU^T (low-rank SPD)
|   |-- manifold.py            # ManifoldProjection + gamma_scale
|   |-- gravity.py             # GravityField: massa + RFF + deformacao de U(x)
|   |-- dimensional_gate.py    # DimensionalGate: dimD(p) variavel
|   |-- layers.py              # RMSNorm, FeedForward (SwiGLU), DRMTransformerBlock
|   |-- losses.py              # 5 losses: metric_reg, diversity, orthogonality, axis_var, anchor_align
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
|   |-- train_distributed.py        # Script de lancamento (single/multi GPU)
|   |-- prepare_multilingual_data.py # Download Wikipedia + tokenize o200k_base + remap
|   |-- extract_drm_vectors.py      # Extrai coords, G_diag, gamma, mass
|   +-- voronoi_foliation_drm.py    # 9 fases: Voronoi, LTSA, Homology, Reeb, ARI
|
|-- configs/scaling/            # 11 configs: 1M, 5M, 10M, 15M, 50M, 350M, 1.3B, 13B, 70B, 162B, 640B
|   +-- multilingual/          # Mesmas configs com vocab_size=50000 (o200k_base subset)
|
|-- eval-results/              # Resultados de foliation e avaliacao
|
|-- empirical/                 # Validacao empirica dos eixos semanticos
|   |-- results.json           # Resultados consolidados
|   |-- figures/               # Plots: PCA, t-SNE, axis heatmap, separation
|   +-- tests/                 # 4 testes: axes_projection, axis_stats, semantic_separation, geometry_correlation
|
|-- docs/
|   |-- process/               # Documentacao de cada mudanca (numerada + datada)
+   +-- scaling/runpod/        # Guia de scaling para RunPod
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
