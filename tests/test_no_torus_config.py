from pathlib import Path

import torch
import yaml

from drm_transformer.config import DRMTransformerConfig
from drm_transformer.model import DRMTransformer


def test_no_torus_config_loads_and_forward_runs():
    path = Path("configs/ablations/no_torus.yaml")
    cfg = yaml.safe_load(path.read_text()) or {}
    model_keys = {
        "vocab_size",
        "max_seq_len",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "dropout",
        "bias",
        "d_manifold",
        "metric_hidden",
        "metric_rank",
        "n_quad",
        "distance_mode",
        "quad_points",
        "distance_chunk_size",
        "n_anchors",
        "gamma_enabled",
        "gamma_c",
        "gamma_alpha",
        "temperature_init",
        "temperature_min",
        "gravity_enabled",
        "gravity_strength",
        "gravity_n_rff",
        "variable_dim",
    }
    model_cfg = {k: v for k, v in cfg.items() if k in model_keys}
    model_cfg.update(
        vocab_size=64,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
        d_manifold=4,
        metric_hidden=8,
        gravity_n_rff=8,
        dropout=0.0,
    )
    config = DRMTransformerConfig(**model_cfg)
    assert cfg.get("lambda_torus", 1.0) == 0.0

    model = DRMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (1, 8, config.vocab_size)
    assert torch.isfinite(logits).all()
