# Guia de Reprodutibilidade

Passo-a-passo para reproduzir resultados do DRM Transformer do zero.

## 1. Ambiente

```bash
# Python 3.12+, CUDA 12.8+
python --version   # 3.12.3
nvidia-smi         # CUDA 12.8

# Clone e setup
git clone https://github.com/gnai-creator/drm_transformer.git
cd drm_transformer

# Ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Dependencias exatas (reprodutibilidade)
pip install -r requirements-lock.txt

# OU dependencias minimas (pode variar)
pip install -e ".[all]"
```

### Versoes Criticas

| Pacote | Versao Testada |
|--------|---------------|
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| cuDNN | 91002 |
| NumPy | 2.4.3 |

## 2. Dados

```bash
# Login no HuggingFace (necessario para CulturaX)
huggingface-cli login

# Download + tokenizacao (streaming, memoria constante)
python scripts/prepare_multilingual_data.py \
    --output-dir data/multilingual \
    --max-tokens 20000000000 \
    --vocab-size 50000 \
    --langs en,pt,es,fr,de
```

Para testes rapidos (minutos em vez de horas):

```bash
python scripts/prepare_multilingual_data.py \
    --source wikipedia \
    --output-dir data/test \
    --max-tokens 10000000 \
    --vocab-size 50000 \
    --langs en
```

## 3. Treinamento

```bash
# Single GPU, seed fixa, modo deterministico
python scripts/train_distributed.py \
    --config configs/scaling/multilingual/15m.yaml \
    --data-dir data/multilingual \
    --seed 42 \
    --deterministic

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train_distributed.py \
    --config configs/scaling/multilingual/350m.yaml \
    --data-dir data/multilingual \
    --seed 42
```

### O que `--seed 42` faz

Fixa todas as fontes de aleatoriedade:
- `random.seed(42)`
- `PYTHONHASHSEED=42`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`

### O que `--deterministic` faz

Ativa flags de determinismo no PyTorch:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`

**Nota:** modo deterministico pode reduzir performance em ~5-10%.

## 4. Run Manifest

A cada treino, um `run_manifest.json` e salvo automaticamente no diretorio
de checkpoints com:

```json
{
  "timestamp": "2026-03-23T...",
  "seed": 42,
  "config_path": "configs/scaling/multilingual/15m.yaml",
  "config": { "...config final resolvida..." },
  "config_hash": "a1b2c3d4e5f6g7h8",
  "git": {
    "commit": "abc123...",
    "branch": "main",
    "dirty": "False"
  },
  "hardware": {
    "hostname": "...",
    "gpu_name": "NVIDIA RTX ...",
    "gpu_count": 1,
    "gpu_memory_gb": 24.0,
    "cpu_count": 16,
    "ram_gb": 64.0
  },
  "dependencies": {
    "python": "3.12.3",
    "torch": "2.10.0+cu128",
    "cuda": "12.8",
    "numpy": "2.4.3"
  }
}
```

## 5. Criterio de Sucesso

Duas execucoes com o mesmo comando, seed e hardware devem produzir:
- **Loss final**: dentro de 1% de tolerancia
- **Checkpoints**: pesos identicos em modo `--deterministic`

Para verificar:

```bash
# Run 1
python scripts/train_distributed.py \
    --config configs/scaling/multilingual/1m.yaml \
    --data-dir data/test --seed 42 --deterministic

# Mover checkpoints
mv checkpoints checkpoints_run1

# Run 2
python scripts/train_distributed.py \
    --config configs/scaling/multilingual/1m.yaml \
    --data-dir data/test --seed 42 --deterministic

# Comparar
python -c "
import torch
a = torch.load('checkpoints_run1/final.pt', weights_only=False)
b = torch.load('checkpoints/final.pt', weights_only=False)
for k in a['model']:
    diff = (a['model'][k] - b['model'][k]).abs().max().item()
    if diff > 0:
        print(f'{k}: max_diff={diff:.2e}')
print('PASS' if all(
    (a['model'][k] - b['model'][k]).abs().max().item() < 1e-6
    for k in a['model']
) else 'FAIL')
"
```

## 6. Limitacoes Conhecidas

- **Multi-GPU**: NCCL pode introduzir nao-determinismo em reducoes.
  Para determinismo total, use single GPU.
- **`torch.use_deterministic_algorithms`**: algumas ops CUDA nao tem
  implementacao deterministica. O `warn_only=True` emite warning em vez
  de erro nestes casos.
- **Hardware diferente**: resultados reproduziveis exigem mesmo modelo
  de GPU. GPUs diferentes (ex: A100 vs RTX 4090) podem dar resultados
  ligeiramente distintos devido a diferencias em FP.
