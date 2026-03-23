# Arte Anterior e Trabalhos Relacionados

Este documento lista a origem dos conceitos implementados no DRM Transformer
e os trabalhos academicos que fundamentam a teoria subjacente.

---

## Conceitos Originais (Felipe Maya Muniz)

Os seguintes conceitos foram **criados, projetados e implementados originalmente
por Felipe Maya Muniz** nos projetos ATIC (Adaptive Turing Intelligent Cognition,
2025-2026) e Aletheion LLM v2 (2025-2026). No DRM Transformer, estes conceitos
sao aplicados diretamente na arquitetura do transformer, substituindo componentes
convencionais por equivalentes geometricos.

| Conceito | Descricao | Origem |
|----------|-----------|--------|
| **DRM (Directional Relational Manifolds)** | Manifold Riemanniano com dimensionalidade variavel D(p), tensor metrico aprendido G(x), geodesicas e curvatura | ATIC v2.0+ (2025) / Papers DRM V1.1, Geometry of Consciousness V1.2 |
| **Geodesic Attention** | Mecanismo de atencao onde a distancia entre Q e K e computada como distancia geodesica sob G(x), substituindo o dot-product Euclidiano | Paper DRM V1.1 (2025) |
| **Gravitational Token Embedding** | Cada token possui massa aprendida que deforma o tensor metrico G(x) na vizinhanca, analogamente a gravidade na Relatividade Geral | Paper DRM Relativistic Dynamics (2025) |
| **Gamma-Scaling (Relativistic Dynamics)** | Fator de Lorentz gamma aplicado como scaling adaptativo na atencao, variando a resolucao conforme a "velocidade" no manifold | Paper DRM Relativistic Dynamics (2025) |
| **DimensionalGate** | Gate aprendido que define dimensionalidade efetiva dimD(p) por token via mascara suave, implementando a dimensionalidade variavel do DRM | Paper DRM V1.1 (2025) |
| **Metric Diversity Loss** | Funcao de perda que penaliza G(x) constante ao longo da sequencia, incentivando curvatura aprendida e evitando colapso para espaco plano | DRM Transformer (2026) |

**Repositorios relacionados:**
- ATIC: [github.com/gnai-creator/atic_consulting](https://github.com/gnai-creator/atic_consulting)
- Aletheion LLM v2: [github.com/gnai-creator/aletheion-llm-v2](https://github.com/gnai-creator/aletheion-llm-v2)

---

## Trabalhos Academicos Relacionados

Os trabalhos abaixo fornecem a fundamentacao teorica e matematica sobre a qual
os conceitos originais se apoiam.

### Arquitetura Transformer

- **Vaswani et al. (2017)** - "Attention Is All You Need"
  Arquitetura Transformer original com multi-head self-attention.
  O DRM Transformer substitui o dot-product attention por geodesic attention
  sob tensor metrico aprendido.

### Geometria Riemanniana em ML

- **Chen et al. (2018)** - "Riemannian Geometry of Deep Generative Models"
  Estudo da geometria Riemanniana em modelos generativos profundos.
  Fundamenta a ideia de que o espaco latente de redes neurais possui
  geometria intrinseca nao-trivial.

- **Ganea et al. (2018)** - "Hyperbolic Neural Networks"
  Redes neurais operando em espacos hiperbolicos. O DRM generaliza esta
  abordagem: em vez de geometria hiperbolica fixa, usa tensor metrico
  G(x) aprendido que pode assumir qualquer curvatura.

### Information Geometry

- **Amari & Nagaoka (2000)** - "Methods of Information Geometry" (AMS)
  Framework canonico que equipa o espaco de distribuicoes de probabilidade
  com estrutura Riemanniana via Fisher Information Matrix.
  **Relacao com DRM:** Amari opera em manifolds de dimensao fixa.
  O DRM generaliza permitindo dim D(p) variar por ponto -- modelando
  sistemas cujos graus de liberdade ativos emergem ou colapsam
  dinamicamente.

### Geodesicas em Representacoes Aprendidas

- **Arvanitidis et al. (2018)** - "Latent Space Oddity: on the Curvature
  of Deep Generative Models"
  Demonstra que geodesicas em espacos latentes capturam melhor a estrutura
  dos dados do que distancias Euclidianas. Fundamenta a motivacao para
  geodesic attention.

### Geometria Hiperbolica

- **Nickel & Kiela (2017)** - "Poincare Embeddings for Learning Hierarchical
  Representations"
  Embeddings em espacos hiperbolicos para hierarquias. O DRM usa geometria
  Riemanniana geral (nao restrita a Poincare) com tensor metrico aprendivel.

---

## Resumo

A distincao fundamental e:
- **Trabalhos academicos acima**: fornecem a teoria matematica e os frameworks gerais
- **Conceitos de Felipe Maya Muniz**: aplicam, estendem e combinam estas teorias
  de formas originais para criar o DRM Transformer

Nenhum dos trabalhos academicos listados implementa Geodesic Attention com
tensor metrico aprendido G(x), Gravitational Token Embedding, Gamma-Scaling
relativístico, DimensionalGate com dimD(p) variavel, ou Metric Diversity Loss.
Estes sao contribuicoes originais de Felipe Maya Muniz.

---

(c) 2026 Felipe Maya Muniz. All rights reserved.
