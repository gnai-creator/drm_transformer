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
+-- config.py                       # DRMTransformerConfig (dataclass, todos os parametros)
+-- model.py                        # DRMTransformerModel (forward principal)
+-- layers.py                       # DRMBlock (Attention + FFN + LayerNorm + Residual)
+-- attention.py                    # GeodesicAttention (distancia sob G(x), multi-head)
+-- metric_net.py                   # MetricNet: G(x) diagonal + low-rank semantic axes
+-- manifold.py                     # Operacoes no manifold (gamma-scaling, coordenadas)
+-- gravity.py                      # GravityField: massa deforma G(x) via RFF kernel
+-- dimensional_gate.py             # DimensionalGate: dimD(p) variavel por token
+-- losses.py                       # Loss composta: CE + metric_reg + metric_diversity
+-- __init__.py
|
+-- training/
|   +-- trainer.py                  # Trainer (single/multi-GPU, mixed precision)
|   +-- distributed.py              # DDP + FSDP setup, DistributedSampler
|   +-- data.py                     # ShardedDataset (.npy/.bin), DataLoader utilities
|   +-- __init__.py
|
+-- evaluation/
    +-- foliation.py                # DRMFoliationEvaluator (Voronoi, LTSA, Homology, Reeb, ARI)
    +-- __init__.py

scripts/
+-- train_distributed.py            # Lancamento de treino (single/multi GPU via torchrun)
+-- prepare_multilingual_data.py    # Download + tokenize + remap (CulturaX/Wikipedia, streaming)
+-- extract_drm_vectors.py          # Extrai coords, G_diag, gamma, mass de checkpoints
+-- voronoi_foliation_drm.py        # 9 fases: Voronoi, LTSA, Homology, Reeb, ARI

empirical/
+-- tests/
|   +-- run_all.py                  # Runner: executa 5 testes + gera figuras
|   +-- test_axis_statistics.py     # Estatisticas dos eixos do manifold
|   +-- test_axes_projection.py     # Projecao dos eixos semanticos
|   +-- test_calibration.py         # ECE, MCE, Brier score, perplexidade
|   +-- test_geometry_correlation.py # Correlacao entre geometria e semantica
|   +-- test_semantic_separation.py # Separacao semantica no espaco aprendido
|   +-- utils.py                    # Utilitarios compartilhados
+-- figures/
    +-- plot_axis_heatmap.py        # Heatmap dos eixos do tensor metrico
    +-- plot_axis_separation.py     # Separacao entre eixos semanticos
    +-- plot_calibration.py         # Reliability diagram + confidence histogram
    +-- plot_combined.py            # Dashboard combinado (ECE, MCE, Brier, PPL)
    +-- plot_pca.py                 # PCA dos embeddings no manifold
    +-- plot_tsne.py                # t-SNE dos embeddings

configs/scaling/multilingual/       # 12 configs de escala (1M a 640B params)
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
gamma_ij = lorentz_factor(d_ij)             # Fator relativistico
attn_weights = softmax(-d_ij * gamma_ij)    # Proximidade geodesica
output = attn_weights @ V
```

Tokens proximos no manifold (distancia geodesica pequena) recebem mais
atencao. A geometria e aprendida end-to-end via MetricNet. A gravidade
dos tokens pesados (alta massa) curva o espaco ao redor, atraindo tokens
vizinhos. O fator gamma escala adaptativamente a resolucao.

## MetricNet: G(x) Diagonal + Low-Rank Semantic Axes

O tensor metrico G(x) e parametrizado como diagonal + low-rank para
eficiencia e expressividade:

```
hidden [B,T,d_model]
    |
    MLP
    |
    v
diag [B,T,d_manifold]          # Componente diagonal (softplus + epsilon)
axes [B,T,metric_rank,d_manifold]  # Eixos semanticos low-rank
    |
    G(x) = diag(diag) + axes^T @ axes   # SPD garantido
```

Propriedades:
- **Diagonal base**: captura escala por dimensao
- **Low-rank axes**: captura correlacoes semanticas (metric_rank << d_manifold)
- **SPD garantido**: diag > 0 + outer product sempre positivo semi-definido
- **d_manifold**: dimensao do manifold (tipicamente 16), independente de d_model

## GravityField: Massa dos Tokens via RFF Kernel

Cada token possui uma massa aprendida. O campo gravitacional usa Random
Fourier Features (RFF) para eficiencia:

```
G_eff(x) = G(x) + gravity_strength * sum_i  m_i * RFF_kernel(x, x_i)
```

Parametros:
- `gravity_strength`: forca do campo (default 0.1)
- `gravity_n_rff`: numero de features aleatorias (default 64)
- `n_anchors`: pontos de ancora para o campo (default 6)

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

## Treinamento Distribuido

Suporte completo a treino multi-GPU via `training/distributed.py`:

- **DDP** (Distributed Data Parallel): gradientes sincronizados entre GPUs
- **FSDP** (Fully Sharded Data Parallel): para modelos grandes (>1B params)
- **Mixed precision** (fp16/bf16): reduz memoria e acelera compute
- **Gradient checkpointing**: troca compute por memoria em modelos profundos
- **DistributedSampler**: particiona dados entre workers

Lancamento via torchrun:
```bash
torchrun --nproc_per_node=4 scripts/train_distributed.py --config configs/scaling/multilingual/350m.yaml
```

## Pipeline de Dados

O `prepare_multilingual_data.py` opera em 2 passes com memoria constante:

1. **Pass 1**: Stream textos (CulturaX ou Wikipedia) -> tokeniza (o200k_base) -> conta freq -> salva shards raw (uint32)
2. **Pass 2**: Constroi vocab mapping top-K -> remapeia shards raw -> salva shards finais (uint16)

Features:
- Checkpoint por lingua (resume com `--resume`)
- CulturaX (default): 6.3T tokens, 167 linguas, download rapido via parquet
- Wikipedia: publico, sem autenticacao
- Tokenizacao: tiktoken o200k_base com remapeamento para vocab compacto (50K)

## Avaliacao Empirica

Suite de 5 testes diagnosticos + 6 visualizacoes:

| Teste | O que mede |
|-------|-----------|
| axis_statistics | Estatisticas dos eixos semanticos do manifold |
| axes_projection | Qualidade da projecao dos eixos low-rank |
| calibration | ECE, MCE, Brier score, perplexidade |
| geometry_correlation | Correlacao entre geometria aprendida e semantica |
| semantic_separation | Separacao de clusters no espaco do manifold |

Foliation evaluator (pipeline de 9 fases): Voronoi tessellation, LTSA,
Homologia persistente, grafo de Reeb, ARI (Adjusted Rand Index).

## Parametros de Configuracao

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| vocab_size | int | 50257 | Tamanho do vocabulario |
| max_seq_len | int | 1024 | Comprimento maximo de sequencia |
| d_model | int | 384 | Dimensao do modelo |
| n_layers | int | 6 | Numero de blocos DRM |
| n_heads | int | 6 | Numero de heads de atencao |
| d_ff | int | 1536 | Dimensao do feed-forward |
| dropout | float | 0.1 | Taxa de dropout |
| bias | bool | False | Bias nas camadas lineares |
| d_manifold | int | 16 | Dimensao do manifold epistemico |
| metric_hidden | int | 64 | Dimensao oculta do MetricNet |
| metric_rank | int | 4 | Rank dos eixos semanticos low-rank |
| n_quad | int | 0 | Pontos de quadratura (0=Mahalanobis local) |
| n_anchors | int | 6 | Pontos de ancora do campo gravitacional |
| gamma_enabled | bool | True | Habilita Lorentz gamma-scaling |
| gamma_c | float | 4.0 | Velocidade limite c para gamma |
| gamma_alpha | float | 0.0 | Alpha do gamma-scaling |
| temperature_init | float | 1.0 | Temperatura inicial da atencao |
| temperature_min | float | 0.5 | Temperatura minima |
| gravity_enabled | bool | True | Habilita campo gravitacional |
| gravity_strength | float | 0.1 | Forca da gravidade |
| gravity_n_rff | int | 64 | Numero de Random Fourier Features |
| variable_dim | bool | True | Habilita DimensionalGate |

## Scaling Configs

12 configuracoes multilingual de 1M a 640B parametros em `configs/scaling/multilingual/`:

| Config | Params | d_model | n_layers | n_heads |
|--------|--------|---------|----------|---------|
| 1m | ~1M | 64 | 4 | 2 |
| 5m | ~5M | 96 | 6 | 3 |
| 10m | ~10M | 160 | 6 | 4 |
| 15m | ~15M | 256 | 6 | 4 |
| 50m | ~50M | 512 | 8 | 8 |
| 125m | ~125M | 768 | 12 | 12 |
| 350m | ~350M | 1024 | 24 | 16 |
| 1.3b | ~1.3B | 2048 | 24 | 16 |
| 13b | ~13B | 5120 | 40 | 40 |
| 70b | ~70B | 8192 | 80 | 64 |
| 162b | ~162B | 12288 | 96 | 96 |
| 640b | ~640B | 16384 | 126 | 128 |

---

(c) 2026 Felipe Maya Muniz. Todos os direitos reservados.
