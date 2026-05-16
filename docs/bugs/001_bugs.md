# Auditoria 001 - Bugs e pendencias

Status atualizado em 2026-05-16.

## Corrigido

- [x] 1. MetricNet nascia "morta".
  - Problema: `MetricNet` zerava a ultima camada. Como a contribuicao na attention e `||U^T delta||^2`, `U=0` tambem zera o gradiente principal para `U`.
  - Correcao: `MetricNet.reset_parameters()` inicializa `U` perto de zero, mas nao exatamente zero.
  - Arquivos: `src/drm_transformer/metric_net.py`, `src/drm_transformer/model.py`.
  - Teste: `test_metric_net_has_attention_gradient`.

- [x] 2. `gamma_enabled: true` nao fazia efeito nos configs principais.
  - Problema: `gamma_alpha=0.0` tornava `effective_gamma=1.0`, anulando o gamma-scaling.
  - Correcao: default de `gamma_alpha` mudou para `1.0`; configs `baseline/full/no_gravity/no_variable_dim` tambem foram atualizados.
  - Arquivos: `src/drm_transformer/config.py`, `configs/baselines/small_3.5M.yaml`, `configs/ablations/*.yaml`.
  - Teste: `test_gamma_alpha_changes_attention_distance`.

- [x] 3. Pipeline de dados carregava shards inteiros em memoria.
  - Problema: `ShardedDataset` concatenava todos os shards em RAM e `train_distributed.py` usava `total_tokens` como limite de carregamento.
  - Correcao: dataset agora usa `np.load(..., mmap_mode="r")` / `np.memmap`, le janelas por indice global e cruza fronteiras de shard quando necessario. O treino usa `data_max_tokens` opcional para limitar leitura, separado de `total_tokens`.
  - Arquivos: `src/drm_transformer/training/data.py`, `scripts/train_distributed.py`.
  - Teste: `test_sharded_dataset_reads_across_shards`.

- [x] 4. `requirements-lock.txt` era congelamento de ambiente, nao do projeto.
  - Problema: continha dependencias sem uso no repo, como `anthropic`, `openai`, `fastapi`, `stripe`, `psycopg2-binary`.
  - Correcao: lock reduzido para runtime, testes, dados e avaliacao. Inclui `-e .[all]`, entao o comando correto e `pip install -r requirements-lock.txt`.
  - Arquivo: `requirements-lock.txt`.

- [x] 5. Smoke tests nao cobriam os bugs centrais da auditoria.
  - Correcao: adicionados testes para gradiente real da MetricNet, efeito do gamma e leitura sharded/memmap.
  - Arquivo: `tests/test_smoke.py`.
  - Validacao local: `.\.venv\Scripts\python.exe -m pytest tests\test_smoke.py -q` passou com `11 passed`.

## Pendente

- [ ] 6. Revalidar resultados empiricos apos as correcoes.
  - Motivo: as metricas atuais em `eval-results/` e `MODEL_CARD.md` foram geradas antes das correcoes de MetricNet/gamma/dataset. Elas nao devem ser tratadas como evidencia final da arquitetura corrigida.
  - Fazer:
    - Rodar baseline `small_3.5M` novamente.
    - Rodar ablations `full`, `no_gamma`, `no_gravity`, `no_variable_dim`.
    - Regenerar `eval-results/`.
    - Atualizar `MODEL_CARD.md` com os novos numeros.

- [ ] 7. Corrigir encoding de documentos.
  - Problema: alguns textos aparecem com mojibake em alguns terminais, por exemplo acentos renderizados incorretamente.
  - Fazer:
    - Normalizar Markdown para UTF-8.
    - Revisar `MODEL_CARD.md`, `README.md`, `ARCHITECTURE.md`, `repro.md` e docs em `docs/process/`.
    - Evitar misturar arquivos salvos em codepage Windows com UTF-8.

- [ ] 8. Melhorar seguranca de carregamento de checkpoints.
  - Problema: varios scripts usam `torch.load(..., weights_only=False)`, o que e aceitavel para checkpoints locais confiaveis, mas ruim como default em scripts publicos.
  - Locais conhecidos:
    - `src/drm_transformer/training/trainer.py`
    - `scripts/train_distributed.py`
    - `scripts/eval_standard.py`
    - `scripts/extract_drm_vectors.py`
  - Fazer:
    - Preferir `weights_only=True` quando carregar apenas pesos.
    - Documentar quando `weights_only=False` for necessario para optimizer/scaler/config.
    - Adicionar aviso claro para checkpoints nao confiaveis.

- [ ] 9. Tornar mixed precision robusto em CPU.
  - Problema: `DRMTrainer` usa `torch.autocast(device_type="cuda", ...)` mesmo que o device seja CPU. Hoje os configs principais usam `mixed_precision: none`, mas isso pode quebrar em uso CPU com `bf16/fp16`.
  - Fazer:
    - Derivar `device_type` de `self.device`.
    - Desabilitar scaler fp16 quando CUDA nao estiver disponivel.
    - Adicionar teste simples de trainer CPU com `mixed_precision: none`.

- [ ] 10. Validar aprendizado da gravidade separadamente.
  - Estado: ha teste de forward sem/com gravidade, mas nao ha teste que confirme que `GravityField.mass_net` recebe gradiente util ou que alterar gravidade muda a attention de forma mensuravel.
  - Fazer:
    - Adicionar teste de gradiente para `mass_net`.
    - Adicionar teste comparando outputs com `gravity_enabled=True/False` sob seed fixa e dropout zero.

- [ ] 11. Dataset sharded ainda e map-style, nao streaming puro.
  - Estado: a memoria foi corrigida com mmap, mas o dataset ainda depende de indice global e `DistributedSampler`.
  - Fazer se escalar para datasets muito grandes/remotos:
    - Avaliar `IterableDataset` por shard.
    - Suportar shuffle por shard + buffer local.
    - Evitar `len()` obrigatorio quando o dataset for stream remoto.

- [ ] 12. Limpar artefatos temporarios inacessiveis no Windows.
  - Contexto: durante testes com `tempfile`, alguns diretorios temporarios ficaram com ACL quebrada no workspace/sandbox.
  - Fazer:
    - Remover manualmente como administrador se ainda aparecerem no `git status`/Explorer.
    - Manter testes usando diretorio controlado em `tests/` com cleanup tolerante.

## Prioridade recomendada

1. Reexecutar baseline + ablations com o modelo corrigido.
2. Atualizar `MODEL_CARD.md` e `eval-results/` com resultados novos.
3. Corrigir encoding dos Markdown.
4. Endurecer `torch.load` e mixed precision.
5. Adicionar testes especificos para gravidade.
