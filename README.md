# DRM Transformer

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Commercial License](https://img.shields.io/badge/License-Commercial-orange.svg)](LICENSE-COMMERCIAL.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![Configs](https://img.shields.io/badge/Scaling-1M%20to%20640B-green.svg)](configs/scaling/)
[![Architecture](https://img.shields.io/badge/Attention-Geodesic-blueviolet.svg)](#inovacoes-principais)
[![Papers](https://img.shields.io/badge/Papers-3%20DRM-yellow.svg)](#referencias)

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
   adaptativamente a resolucao da atencao conforme a "velocidade" no manifold.

---

## Arquitetura

```
 input_ids [B, T]
      |
      v
 +------------------+
 |   Embed (token)  |
 +------------------+
      |
      v
 +------------------+
 | DimensionalGate  |  --> dimD(p) por token
 +------------------+
      |
      v
 +==========================================+
 |  DRM Block x N                           |
 |                                          |
 |  DRM Attention:                          |
 |    G(x) <- MetricNet(hidden)             |
 |    gravity <- GravityField(mass, G)      |
 |    dist <- geodesic_dist(Q, K, G)        |
 |    gamma <- lorentz_factor(dist)         |
 |    attn = softmax(-dist * gamma) @ V     |
 |                                          |
 |  FFN (GELU/SwiGLU) + Residual           |
 +==========================================+
      |
      v
 +------------------+
 |     LM Head      |
 +------------------+
      |
      v
  logits [B, T, V]
```

---

## Quick Start

### Instalacao

```bash
git clone https://github.com/gnai-creator/drm-transformer.git
cd drm-transformer
pip install -e ".[dev]"
```

### Treinamento

```bash
# Single GPU
python scripts/train.py \
    --config configs/350m.yaml \
    --data-dir data/fineweb

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/350m.yaml \
    --data-dir data/fineweb
```

### Geracao

```python
import torch
from drm_transformer import DRMTransformerConfig, DRMTransformerModel

config = DRMTransformerConfig()
model = DRMTransformerModel(config)

input_ids = torch.randint(0, config.vocab_size, (1, 128))
output = model(input_ids)

logits = output.logits  # [1, 128, 50257]
```

---

## Papers

Baseado em tres papers de Felipe Maya Muniz:

1. **DRM V1.1** - Directional Relational Manifolds: geometria com
   dimensionalidade variavel e tensor metrico aprendido
2. **Geometry of Consciousness V1.2** - Geometria de campos conscientes
   no manifold DRM
3. **DRM Relativistic Dynamics** - Dinamica relativistica com fator
   de Lorentz e campo gravitacional de tokens

---

## Estrutura do Projeto

```
drm-transformer/
|-- LICENSE                     # AGPL-3.0 (texto completo)
|-- LICENSE-COMMERCIAL.md       # Termos da licenca comercial
|-- CLA.md                      # Contributor License Agreement
|-- ARCHITECTURE.md             # Arquitetura detalhada
|-- PRIOR_ART.md                # Arte anterior e trabalhos relacionados
|-- ROADMAP.md                  # Plano de desenvolvimento
|-- README.md                   # Este arquivo
|
|-- src/drm_transformer/
|   |-- __init__.py
|   |-- config.py               # DRMTransformerConfig
|   |
|   |-- model/                  # Nucleo do modelo
|   |   |-- model.py            # DRMTransformerModel (forward principal)
|   |   |-- embeddings.py       # TokenEmbedding
|   |   +-- output.py           # ModelOutput dataclass
|   |
|   |-- attention/              # DRM Attention
|   |   |-- geodesic_attention.py # GeodesicAttention
|   |   +-- gamma_scaling.py    # Lorentz gamma-factor
|   |
|   |-- metric_net/             # Tensor metrico aprendido
|   |   |-- metric_net.py       # MetricNet: G(x) via MLP + Cholesky
|   |   +-- cholesky_param.py   # Parametrizacao SPD
|   |
|   |-- manifold/               # Operacoes no manifold
|   |   |-- geodesic_distance.py # Distancia geodesica
|   |   |-- christoffel.py      # Simbolos de Christoffel
|   |   +-- curvature.py        # Curvatura de Ricci
|   |
|   |-- gravity/                # Campo gravitacional
|   |   |-- gravity_field.py    # GravityField
|   |   +-- token_mass.py       # Massa por token
|   |
|   |-- dimensional_gate/       # Dimensionalidade variavel
|   |   |-- dimensional_gate.py # DimensionalGate: dimD(p)
|   |   +-- soft_mask.py        # Mascara suave
|   |
|   |-- layers/                 # Blocos do transformer
|   |   |-- drm_block.py        # DRMBlock
|   |   |-- feed_forward.py     # FFN (GELU/SwiGLU)
|   |   +-- lm_head.py          # Language model head
|   |
|   |-- losses/                 # Funcoes de perda
|   |   |-- composite_loss.py   # CE + metric_reg + metric_diversity
|   |   |-- metric_regularization.py
|   |   +-- metric_diversity.py
|   |
|   |-- training/               # Pipeline de treinamento
|   |   |-- trainer.py
|   |   |-- data.py
|   |   +-- scheduler.py
|   |
|   +-- inference/              # Geracao
|       +-- generator.py
|
|-- tests/                      # Testes unitarios
|-- configs/                    # Configs YAML por escala
|-- scripts/                    # Scripts de treinamento e avaliacao
|-- checkpoints/                # Checkpoints de modelos
+-- docs/process/               # Documentacao de processos
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
  url = {https://github.com/gnai-creator/drm-transformer},
  license = {AGPL-3.0-or-later}
}
```

---

Copyright (C) 2026 Felipe Maya Muniz
