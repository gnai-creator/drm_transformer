# 003 - DRM Marco A: Manifold Attention Tensor Anatomy

**Status:** planejado / design documentado  
**Origem:** ideias de tensor networks e truncated-SVD inspiradas por ITensors.jl / ITensorMPS.jl  
**Escopo:** diagnostico offline do DRM Transformer, sem alterar o treinamento principal

## 1. Objetivo

O DRM Marco A propõe uma análise tensorial offline da Geodesic Attention do DRM Transformer.

A pergunta central é:

```text
A atenção geométrica do DRM Transformer tem estrutura de baixa dimensão, baixo rank,
setores redundantes ou padrões de compressibilidade por layer/head/token?
```

Se a resposta for positiva, essa estrutura pode orientar:

```text
- compressão de MetricNet / GravityField / attention logits;
- seleção de layers/heads que merecem crescimento/adaptação;
- roteamento posterior via SAINT-G;
- pruning de heads ou dimensões do manifold;
- diagnóstico de estabilidade geométrica;
- desenho de kernels de inferência mais baratos;
- comparação mais honesta entre DRM e attention padrão.
```

Este marco é inspirado por ITensors.jl e ITensorMPS.jl, mas deve ser implementado em PyTorch no repositório `drm_transformer`. Não há recomendação de adicionar Julia como dependência do runtime.

## 2. Contexto técnico

O bloco de atenção DRM atual computa, por bloco/layer:

```text
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
q, k = RoPE(q), RoPE(k)
q_m = sigmoid(q_to_manifold(q))
k_m = sigmoid(k_to_manifold(k))
U(q_m) = MetricNet(q_m)
U_grav = GravityField.deform_U(U, q_m, mass(q_m))
delta = q_m - k_m
dist^2 = ||delta||^2 + ||U^T delta||^2
attn_logits = -dist^2 / temperature
attn_probs = softmax(causal_mask(attn_logits))
out = attn_probs @ v
```

Os tensores de interesse aparecem naturalmente em formatos de alta ordem:

```text
q_m, k_m:       [batch, head, token, d_manifold]
U:              [batch, head, token, d_manifold, metric_rank]
mass:           [batch, head, token, 1]
dist_sq:        [batch, head, query_token, key_token]
attn_logits:    [batch, head, query_token, key_token]
attn_probs:     [batch, head, query_token, key_token]
```

A ideia tensor-network é tratar esses objetos não como arrays opacos, mas como tensores com eixos semânticos explícitos:

```text
batch x layer x head x token x manifold_dim x metric_rank
```

Isso permite medir rank efetivo, espectros singulares, compressibilidade e redundância por eixo.

## 3. Relação com ITensors.jl / ITensorMPS.jl

ITensors.jl fornece uma disciplina de trabalho útil:

```text
- índices nomeados/taggeados em vez de eixos anônimos;
- contrações por identidade de índice;
- fatorações SVD/QR/eigen em agrupamentos arbitrários de índices;
- truncated SVD com cutoff, maxdim, mindim e erro explícito;
- custo de sequência de contração;
- tensores block-sparse/QN.
```

ITensorMPS.jl fornece a família MPS/MPO/DMRG/truncate!, útil como inspiração para decompor grandes tensores em redes de baixo bond dimension.

Para o DRM Transformer, o aproveitamento prático é:

```text
ITensor idea -> PyTorch diagnostic -> metric/report artifact
```

Não é recomendado usar ITensors.jl diretamente no pipeline atual porque isso adicionaria runtime Julia, conversões cross-language e complexidade de GPU sem necessidade imediata.

## 4. Hipóteses

### H1 - Attention logits são low-rank por head/layer

Os logits geométricos podem ter rank efetivo menor do que `seq_len`, especialmente após causal mask e gamma-scaling.

Sinal esperado:

```text
effective_rank(attn_logits[layer, head]) << seq_len
```

Implicação:

```text
possível compressão de logits, heads redundantes, ou aproximação low-rank para inferência.
```

### H2 - Coordenadas no manifold ocupam subespaços setorizados

As coordenadas `q_m` e `k_m` podem usar apenas parte de `d_manifold` para muitos tokens/heads.

Sinal esperado:

```text
rank([batch * token, head * d_manifold]) baixo ou com espectro concentrado
```

Implicação:

```text
DimensionalGate e MetricNet podem ser analisados juntos para identificar dimensões ativas reais.
```

### H3 - Eixos low-rank de MetricNet têm redundância

MetricNet usa `metric_rank`, mas nem todos os eixos podem estar carregando informação útil.

Sinal esperado:

```text
U[..., d_manifold, metric_rank] com colunas correlacionadas ou energia concentrada em poucos eixos
```

Implicação:

```text
metric_rank pode ser reduzido ou regularizado por diversidade/ortogonalidade mais forte.
```

### H4 - GravityField altera rank ou entropia da atenção

A deformação gravitacional pode aumentar ou reduzir a estrutura efetiva de `U` e `dist_sq`.

Sinal esperado:

```text
rank/entropy com gravity_enabled=True diferente de no_gravity ablation
```

Implicação:

```text
gravity deve ser mantida se cria estrutura útil; caso contrário, pode ser custo sem benefício.
```

### H5 - Camadas/heads relevantes têm assinatura tensorial distinta

Layers/heads úteis podem ter:

```text
- maior rank efetivo;
- menor entropia por head;
- maior anisotropia em U;
- maior correlação com anchors;
- atenção mais setorizada por token.
```

Implicação:

```text
SAINT-G pode usar essas assinaturas para escolher onde crescer/adaptar.
```

## 5. Script proposto

Criar:

```text
scripts/analyze_manifold_attention_tensor_anatomy.py
```

Comando Linux recomendado:

```bash
python \
  scripts/analyze_manifold_attention_tensor_anatomy.py \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --config configs/scaling/multilingual/5m.yaml \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --batches 16 \
  --seq-len 128 \
  --output-dir runs/drm_marco_a_manifold_attention_tensor_anatomy
```

O script deve ser offline/diagnóstico e não deve modificar checkpoints.

## 6. Instrumentação mínima

A implementação mais segura é usar hooks ou um modo `return_diagnostics=True` sem alterar o caminho padrão de treino.

Capturar por layer/head:

```text
q_manifold
k_manifold
U_before_gravity
U_after_gravity
mass
dist_euc
dist_lr
dist_sq_before_gamma
dist_sq_after_gamma
attn_logits
attn_probs
```

Se a captura completa de `attn_probs` for muito pesada, usar amostragem:

```text
- primeiras N batches;
- máximo seq_len 128 no primeiro marco;
- subconjunto de layers, e.g. 0, middle, last;
- opcionalmente top-k heads por variância/entropia.
```

## 7. Views tensoriais a analisar

### View A - Coordenadas por token/head

```text
q_m_view = q_manifold.reshape(batch * token, head * d_manifold)
k_m_view = k_manifold.reshape(batch * token, head * d_manifold)
```

Mede se as coordenadas do manifold usam todo o espaço ou subespaços efetivos.

Métricas:

```text
- singular_values;
- effective_rank_99;
- effective_rank_999;
- stable_rank;
- entropy_effective_rank;
- explained_energy_by_top_k.
```

### View B - Eixos low-rank de MetricNet

```text
U_view = U.reshape(batch * head * token, d_manifold, metric_rank)
```

Agrupamentos úteis:

```text
[d_manifold, metric_rank]
[batch * token, d_manifold * metric_rank]
[layer * head, d_manifold * metric_rank]
```

Métricas:

```text
- norma por eixo rank;
- correlação entre colunas de U;
- ortogonalidade média U^T U;
- energia por eixo;
- rank efetivo da matriz achatada.
```

### View C - Attention logits/probs por head

```text
attn_logits_view = attn_logits[layer, head, batch, query_token, key_token]
```

Métricas:

```text
- rank efetivo por [query_token, key_token];
- entropia da distribuição de atenção por query;
- diagonalidade/localidade causal;
- concentração top-k;
- similaridade de heads dentro da mesma layer;
- similaridade da mesma head entre layers.
```

### View D - Distância euclidiana vs low-rank vs gamma

Comparar:

```text
dist_euc
dist_lr
dist_sq_before_gamma
dist_sq_after_gamma
```

Métricas:

```text
- contribuição relativa dist_lr / (dist_euc + dist_lr);
- efeito multiplicativo médio do gamma;
- correlação entre gamma e entropia de atenção;
- mudança no rank efetivo antes/depois do gamma.
```

### View E - Anchors e massa

```text
mass: [batch, head, token, 1]
anchor_distance: [batch, token, n_anchors]
```

Métricas:

```text
- distribuição de massa por token/head;
- correlação massa vs entropia de atenção;
- correlação distância ao anchor vs gamma;
- especialização por anchor;
- tokens que consistentemente deformam a métrica.
```

## 8. Métricas de rank/compressibilidade

Para uma matriz `M` com valores singulares `sigma_i`:

```text
total_energy = sum_i sigma_i^2
energy_k = sum_{i<=k} sigma_i^2 / total_energy
relative_truncation_error(k) = 1 - energy_k
stable_rank = ||M||_F^2 / ||M||_2^2
entropy_effective_rank = exp(-sum_i p_i log p_i), p_i = sigma_i / sum_j sigma_j
```

Registrar ranks mínimos para:

```text
99.0% energia
99.9% energia
99.99% energia
```

Isso segue a disciplina de truncated-SVD inspirada por ITensors.jl, mas implementada com `torch.linalg.svdvals` ou `torch.linalg.svd`.

## 9. Outputs esperados

Diretório de saída:

```text
runs/drm_marco_a_manifold_attention_tensor_anatomy/
```

Arquivos:

```text
summary.json
rank_summary.csv
layer_head_metrics.jsonl
singular_spectra.jsonl
attention_entropy.jsonl
gravity_gamma_metrics.json
results.md
```

Campos mínimos em `summary.json`:

```json
{
  "marco": "DRM_A_manifold_attention_tensor_anatomy",
  "checkpoint": "...",
  "config": "...",
  "batches": 16,
  "seq_len": 128,
  "device": "cuda",
  "layers_analyzed": [0, 3, 5],
  "mean_attention_effective_rank_99": 0,
  "mean_metric_U_effective_rank_99": 0,
  "mean_attention_entropy": 0.0,
  "gravity_rank_delta": 0.0,
  "verdict": "pending"
}
```

## 10. Relatório `results.md`

O relatório deve responder operacionalmente:

```text
1. Quais layers/heads têm menor rank efetivo?
2. Quais layers/heads parecem redundantes?
3. MetricNet usa todos os eixos de metric_rank?
4. GravityField muda rank/entropia de forma útil?
5. Gamma-scaling aumenta resolução onde esperado?
6. Há compressibilidade suficiente para justificar uma otimização?
7. Quais layers/heads seriam candidatos para crescimento/adaptação via SAINT-G?
```

## 11. Critérios de sucesso

DRM Marco A é útil se produzir pelo menos uma destas conclusões acionáveis:

```text
- identificar heads/layers redundantes;
- mostrar que attention logits/probs são compressíveis;
- revelar que U usa menos rank do que metric_rank configurado;
- mostrar que gravity/gamma alteram significativamente a estrutura tensorial;
- sugerir layers/heads para SAINT-G grafting;
- justificar redução de metric_rank, d_manifold ou n_heads;
- justificar uma versão otimizada da atenção geométrica.
```

## 12. Critérios de falha

```text
- todos os espectros são planos e pouco interpretáveis;
- o custo de captura é alto demais;
- as métricas não diferenciam full vs no_gravity/no_gamma/no_variable_dim;
- resultados variam demais entre batches;
- não há recomendação prática para compressão, routing ou arquitetura.
```

## 13. Relação com SAINT-G

DRM Marco A é complementar ao SAINT-G Phase 16:

```text
SAINT-G 4M/4N:
  usa score NTK-style para decidir onde grafts podem ser úteis.

SAINT-G 4O-lite:
  analisa se grafts aceitos são compressíveis por SVD.

DRM Marco A:
  analisa a estrutura interna da atenção geométrica para descobrir onde o
  próprio DRM Transformer tem redundância, rank baixo ou heads/layers críticos.
```

Possível integração futura:

```text
candidate_priority(layer/head) =
    alpha * ntk_sensitivity
  + beta  * attention_rank_anomaly
  + gamma * metric_U_energy
  - delta * redundancy_score
```

Essa prioridade poderia alimentar o roteamento de SAINT-G ou uma rotina interna de pruning/crescimento do DRM Transformer.

## 14. Priorização recomendada

```text
1. Não interromper treinos longos já em execução.
2. Implementar DRM Marco A como análise offline em checkpoint 5M/125M primeiro.
3. Rodar em batches pequenos, seq_len 128, layers 0/middle/last.
4. Comparar full vs no_gravity/no_gamma quando checkpoints existirem.
5. Só depois considerar otimizações arquiteturais ou pruning.
```

## 15. Prior art a citar

ITensor / ITensors.jl:

```bibtex
@article{ITensor,
  title={{The ITensor Software Library for Tensor Network Calculations}},
  author={Matthew Fishman and Steven R. White and E. Miles Stoudenmire},
  journal={SciPost Phys. Codebases},
  pages={4},
  year={2022},
  publisher={SciPost},
  doi={10.21468/SciPostPhysCodeb.4},
  url={https://scipost.org/10.21468/SciPostPhysCodeb.4}
}
```

Repositories:

```text
https://github.com/ITensor/ITensors.jl
https://github.com/ITensor/ITensorMPS.jl
```

NTK-Mirror, for the later SAINT-G/DRM routing bridge:

```bibtex
@software{chlon2026ntkmirror,
  author       = {Leon Chlon},
  title        = {{NTK-Mirror: LoRA-free forward-pass fine-tuning via signed log-mask controllers}},
  year         = {2026},
  organization = {Hassana Labs},
  url          = {https://github.com/leochlon/ntkmirror}
}
```
