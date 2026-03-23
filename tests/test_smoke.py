"""Smoke tests: import, forward pass, config consistency.

Roda rapido (~5s) e valida que o modelo funciona end-to-end.

Uso:
    pytest tests/test_smoke.py -v
    python tests/test_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml


def test_imports():
    """Verifica que todos os modulos importam sem erro."""
    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer
    from drm_transformer.metric_net import MetricNet
    from drm_transformer.gravity import GravityField
    from drm_transformer.dimensional_gate import DimensionalGate
    from drm_transformer.attention import DRMAttention
    from drm_transformer.losses import metric_regularization, metric_diversity_loss
    from drm_transformer.training.trainer import DRMTrainer
    from drm_transformer.training.data import ShardedDataset
    from drm_transformer.training.reproducibility import set_seed, build_run_manifest
    assert True


def test_forward_pass():
    """Verifica forward pass completo (CPU, batch pequeno)."""
    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer

    config = DRMTransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        d_manifold=4,
        metric_hidden=8,
        metric_rank=2,
        gravity_n_rff=8,
        n_anchors=2,
    )
    model = DRMTransformer(config)
    model.eval()

    input_ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 16, 100), f"Expected (2,16,100), got {logits.shape}"
    assert loss is not None
    assert loss.item() > 0
    assert torch.isfinite(loss), "Loss is not finite"


def test_forward_no_gravity():
    """Forward pass sem gravity."""
    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer

    config = DRMTransformerConfig(
        vocab_size=100, d_model=32, n_heads=2, n_layers=2,
        d_ff=64, max_seq_len=16, d_manifold=4, metric_hidden=8,
        gravity_enabled=False,
    )
    model = DRMTransformer(config)
    input_ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits, _ = model(input_ids)
    assert logits.shape == (1, 8, 100)


def test_forward_no_gamma():
    """Forward pass sem gamma-scaling."""
    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer

    config = DRMTransformerConfig(
        vocab_size=100, d_model=32, n_heads=2, n_layers=2,
        d_ff=64, max_seq_len=16, d_manifold=4, metric_hidden=8,
        gamma_enabled=False,
    )
    model = DRMTransformer(config)
    input_ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits, _ = model(input_ids)
    assert logits.shape == (1, 8, 100)


def test_forward_no_variable_dim():
    """Forward pass sem DimensionalGate."""
    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer

    config = DRMTransformerConfig(
        vocab_size=100, d_model=32, n_heads=2, n_layers=2,
        d_ff=64, max_seq_len=16, d_manifold=4, metric_hidden=8,
        variable_dim=False,
    )
    model = DRMTransformer(config)
    input_ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits, _ = model(input_ids)
    assert logits.shape == (1, 8, 100)


def test_generate():
    """Verifica geracao autoregressiva."""
    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer

    config = DRMTransformerConfig(
        vocab_size=100, d_model=32, n_heads=2, n_layers=2,
        d_ff=64, max_seq_len=16, d_manifold=4, metric_hidden=8,
    )
    model = DRMTransformer(config)
    model.eval()

    input_ids = torch.randint(0, 100, (1, 4))
    output = model.generate(input_ids, max_new_tokens=5)
    assert output.shape == (1, 9), f"Expected (1,9), got {output.shape}"


def test_config_consistency():
    """Verifica que configs YAML sao consistentes com DRMTransformerConfig."""
    from drm_transformer.config import DRMTransformerConfig
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(DRMTransformerConfig)}

    # Campos de treino (nao do modelo) que podem existir no YAML
    training_fields = {
        "batch_size", "learning_rate", "warmup_steps", "total_tokens",
        "mixed_precision", "gradient_accumulation_steps", "eval_interval",
        "early_stop_patience", "early_stopping_patience",
        "save_dir", "save_interval", "save_total_limit",
        "log_interval", "data_dir", "eval_data_dir", "lambda_metric_reg",
        "lambda_metric_diversity", "lambda_ortho", "metric_diversity_warmup_steps",
        "target_metric_var", "weight_decay", "adam_beta1", "adam_beta2",
        "max_grad_norm", "min_lr_ratio",
        "gradient_checkpointing", "distributed", "fsdp", "fsdp_sharding",
        "compile_model",
    }

    config_dirs = [
        Path("configs/baselines"),
        Path("configs/ablations"),
        Path("configs/scaling/multilingual"),
    ]

    errors = []
    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        for yaml_path in config_dir.glob("*.yaml"):
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f) or {}

            for key in cfg:
                if key not in valid_fields and key not in training_fields:
                    errors.append(f"{yaml_path}: campo desconhecido '{key}'")

    assert not errors, "Campos invalidos em configs:\n" + "\n".join(errors)


def test_seed_determinism():
    """Verifica que set_seed produz resultados deterministicos."""
    from drm_transformer.training.reproducibility import set_seed
    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer

    config = DRMTransformerConfig(
        vocab_size=50, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, max_seq_len=8, d_manifold=4, metric_hidden=8,
    )

    # Input fixo (nao depende de seed)
    input_ids = torch.tensor([[1, 5, 10, 20, 30, 40, 2, 3]])

    set_seed(42)
    m1 = DRMTransformer(config)
    m1.eval()
    with torch.no_grad():
        logits1, _ = m1(input_ids)

    set_seed(42)
    m2 = DRMTransformer(config)
    m2.eval()
    with torch.no_grad():
        logits2, _ = m2(input_ids)

    assert torch.allclose(logits1, logits2, atol=1e-6), "Seed determinism failed"


if __name__ == "__main__":
    tests = [
        test_imports,
        test_forward_pass,
        test_forward_no_gravity,
        test_forward_no_gamma,
        test_forward_no_variable_dim,
        test_generate,
        test_config_consistency,
        test_seed_determinism,
    ]

    passed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
