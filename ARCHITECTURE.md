# Arquitetura do DRM Transformer

## Visao Geral

O DRM Transformer e um decoder-only Transformer onde o espaco de embeddings
vive num Directional Relational Manifold (DRM). Diferente de Transformers
convencionais que operam em espaco Euclidiano plano, este modelo equipa cada
token com uma geometria Riemanniana aprendida: o tensor metrico G(x) varia
por posicao, tokens possuem massa que deforma a metrica (gravidade), e a
dimensionalidade efetiva D(p) varia por token via DimensionalGate.

A atencao padrao (dot-product) e substituida por **Geodesic Attention**:
a distancia entre queries e keys e computada sob G(x), e o fator de escala
segue a dinamica relativistica (gamma-scaling do fator de Lorentz).

Baseado em tres papers de Felipe Maya Muniz:
- **DRM V1.1** - Directional Relational Manifolds com dimensionalidade variavel
- **Geometry of Consciousness V1.2** - Geometria de campos conscientes no manifold
- **DRM Relativistic Dynamics** - Dinamica relativistica com fator de Lorentz

## Estrutura de Diretorios

```
src/drm_transformer/
|
+-- config.py                    # DRMTransformerConfig (parametros do modelo)
|
+-- model/                       # Nucleo do modelo
|   +-- model.py                 # DRMTransformerModel (forward principal)
|   +-- embeddings.py            # TokenEmbedding + positional encoding
|   +-- output.py                # ModelOutput dataclass
|
+-- attention/                   # DRM Attention
|   +-- geodesic_attention.py    # GeodesicAttention (distancia sob G(x))
|   +-- gamma_scaling.py         # Lorentz gamma-factor adaptive scaling
|
+-- metric_net/                  # Tensor metrico aprendido
|   +-- metric_net.py            # MetricNet: G(x) via MLP + Cholesky (SPD)
|   +-- cholesky_param.py        # Parametrizacao Cholesky para G = LL^T
|
+-- manifold/                    # Operacoes no manifold
|   +-- geodesic_distance.py     # Distancia geodesica sob G(x)
|   +-- christoffel.py           # Simbolos de Christoffel (conexao)
|   +-- curvature.py             # Curvatura de Ricci / escalar
|
+-- gravity/                     # Campo gravitacional
|   +-- gravity_field.py         # GravityField: massa dos tokens deforma G(x)
|   +-- token_mass.py            # Aprendizado de massa por token
|
+-- dimensional_gate/            # Dimensionalidade variavel
|   +-- dimensional_gate.py      # DimensionalGate: dimD(p) por token
|   +-- soft_mask.py             # Mascara suave para dimensoes ativas
|
+-- layers/                      # Blocos do transformer
|   +-- drm_block.py             # DRMBlock (DRMAttention + FFN + LayerNorm)
|   +-- feed_forward.py          # FeedForward (GELU ou SwiGLU)
|   +-- lm_head.py               # Language model head
|
+-- losses/                      # Funcoes de perda
|   +-- composite_loss.py        # Agregador: CE + metric_reg + metric_diversity
|   +-- metric_regularization.py # Penaliza condition number de G(x)
|   +-- metric_diversity.py      # Penaliza G(x) constante (flat space)
|
+-- training/                    # Pipeline de treinamento
|   +-- trainer.py               # Trainer (single-GPU / DDP)
|   +-- data.py                  # DataLoader utilities
|   +-- scheduler.py             # WarmupCosine scheduler
|
+-- inference/                   # Geracao
    +-- generator.py             # Generator (top-k, top-p sampling)
```

## Fluxo de Forward Pass

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
 | DimensionalGate  |  --> dimD(p) por token, mascara dimensoes
 +------------------+
      |
      v
 +==========================================+
 |  DRM Block x N                           |
 |                                          |
 |  +------------------------------------+  |
 |  | DRM Attention                      |  |
 |  |  Q, K, V <- LayerNorm(hidden)      |  |
 |  |  G(x) <- MetricNet(hidden)         |  |
 |  |  gravity <- GravityField(mass, G)  |  |
 |  |  dist <- geodesic_dist(Q, K, G)    |  |
 |  |  gamma <- lorentz_factor(dist)     |  |
 |  |  attn = softmax(-dist * gamma)     |  |
 |  |  out = attn @ V                    |  |
 |  +------------------------------------+  |
 |       |                                  |
 |       v                                  |
 |  +------------------------------------+  |
 |  | FFN (GELU/SwiGLU) + Residual       |  |
 |  +------------------------------------+  |
 |                                          |
 +==========================================+
      |
      v
 +------------------+
 |     LN final     |
 +------------------+
      |
      v
 +------------------+
 |     LM Head      |
 +------------------+
      |
      v
  logits [B, T, V]
```

## DRM Attention vs Atencao Padrao

### Atencao padrao (Vaswani et al.)

```
attn_weights = softmax(Q @ K^T / sqrt(d_k))
output = attn_weights @ V
```

A distancia entre tokens e medida pelo produto escalar em espaco Euclidiano
plano. Todos os tokens vivem na mesma geometria.

### DRM Attention (Geodesic Attention)

```
G_x = MetricNet(hidden)                    # Tensor metrico G(x) [B,T,d,d]
G_x = G_x + gravity_deformation(mass, G)   # Gravidade deforma a metrica
d_ij = geodesic_distance(Q_i, K_j, G_x)    # Distancia sob G(x)
gamma_ij = lorentz_factor(d_ij)             # Fator relativístico
attn_weights = softmax(-d_ij * gamma_ij)    # Proximidade geodesica
output = attn_weights @ V
```

Tokens proximos no manifold (distancia geodesica pequena) recebem mais
atencao. A geometria e aprendida end-to-end via MetricNet. A gravidade
dos tokens pesados (alta massa) curva o espaco ao redor, atraindo tokens
vizinhos. O fator gamma escala adaptativamente a resolucao.

## MetricNet: G(x) via MLP + Cholesky

O tensor metrico G(x) deve ser Simetrico Positivo Definido (SPD) para
definir uma geometria Riemanniana valida. A parametrizacao e:

```
hidden [B,T,d_model]
    |
    MLP (tanh activation)   # C1 suave para Christoffel
    |
    v
L [B,T,d,d]  (triangular inferior)
    |
    G = L @ L^T + epsilon * I   # SPD garantido
```

Propriedades:
- **Tanh activation**: G(x) deve ser C1 para simbolos de Christoffel
- **softplus + epsilon no diagonal**: estabilidade numerica
- **Zero init na ultima camada**: G(x) ~ I na inicializacao
- **LR separado (10x)**: sinal de gradiente fraco relativo ao modelo principal

## GravityField: Massa dos Tokens Deforma a Metrica

Cada token possui uma massa aprendida m_i. Tokens com alta massa deformam
o tensor metrico na sua vizinhanca, analogamente a gravidade na Relatividade
Geral:

```
G_eff(x) = G(x) + sum_i  m_i * K(x, x_i)
```

Onde K(x, x_i) e um kernel de influencia (e.g., Gaussiano) que decai com
a distancia. Tokens "importantes" (alta massa) curvam o espaco, atraindo
a atencao de tokens vizinhos.

## DimensionalGate: Dimensionalidade Variavel dimD(p)

Nem todos os tokens precisam da mesma dimensionalidade. O DimensionalGate
aprende uma mascara suave sobre as dimensoes do embedding:

```
gate_logits = MLP(hidden)       # [B, T, d_model]
gate = sigmoid(gate_logits)     # [0, 1] por dimensao
hidden_gated = hidden * gate    # Dimensoes inativas -> ~0
dimD(p) = sum(gate > threshold) # Dimensionalidade efetiva
```

Tokens simples (artigos, pontuacao) usam poucas dimensoes. Tokens complexos
(termos tecnicos, ambiguos) usam mais dimensoes. Isso implementa a
dimensionalidade variavel D(p) do paper DRM V1.1.

## Composicao de Loss

```
L_total = lambda_ce * CE(logits, targets)
        + lambda_metric_reg * metric_regularization(G)
        + lambda_metric_div * metric_diversity(G)
```

| Componente | Objetivo |
|------------|----------|
| CE | Cross-entropy padrao (next-token prediction) |
| metric_reg | Penaliza condition number alto de G(x) -- estabilidade |
| metric_div | Penaliza G(x) constante -- incentiva curvatura aprendida |

### metric_regularization

Baseada na norma Frobenius da diferenca entre G(x) e a identidade,
ponderada pelo condition number. Impede que G(x) degenere (eigenvalues
muito grandes ou pequenos).

### metric_diversity

Penaliza variancia baixa de G(x) ao longo da sequencia. Se G(x) e
identico para todos os tokens, a geometria e efetivamente plana e o
DRM nao agrega valor. Esta loss incentiva variacao estrutural.

## Parametros de Configuracao

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| d_model | int | 768 | Dimensao do modelo |
| n_heads | int | 12 | Numero de heads de atencao |
| n_layers | int | 12 | Numero de blocos DRM |
| d_ff | int | 3072 | Dimensao do feed-forward |
| vocab_size | int | 50257 | Tamanho do vocabulario |
| max_seq_len | int | 1024 | Comprimento maximo de sequencia |
| dropout | float | 0.1 | Taxa de dropout |
| metric_net_hidden | int | 64 | Dimensao oculta do MetricNet |
| metric_net_activation | str | "tanh" | Ativacao do MetricNet (C1) |
| gravity_kernel | str | "gaussian" | Kernel do campo gravitacional |
| gravity_sigma | float | 1.0 | Largura do kernel gravitacional |
| dim_gate_threshold | float | 0.5 | Threshold do DimensionalGate |
| lambda_ce | float | 1.0 | Peso da loss CE |
| lambda_metric_reg | float | 0.01 | Peso da regularizacao metrica |
| lambda_metric_div | float | 0.001 | Peso da diversidade metrica |
| gamma_scaling | bool | True | Habilita Lorentz gamma-scaling |
| use_gravity | bool | True | Habilita campo gravitacional |
| use_dim_gate | bool | True | Habilita DimensionalGate |

---

(c) 2026 Felipe Maya Muniz. Todos os direitos reservados.
