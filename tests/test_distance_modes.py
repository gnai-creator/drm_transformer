import torch

from drm_transformer.attention import DRMAttention
from drm_transformer.config import DRMTransformerConfig
from drm_transformer.metric_net import MetricNet


def _config(distance_mode: str, quad_points: int = 0) -> DRMTransformerConfig:
    return DRMTransformerConfig(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
        d_manifold=4,
        metric_hidden=8,
        metric_rank=2,
        dropout=0.0,
        gamma_enabled=False,
        gravity_enabled=False,
        variable_dim=False,
        distance_mode=distance_mode,
        quad_points=quad_points,
        distance_chunk_size=2,
    )


def test_local_mode_runs_and_records_finite_distances():
    config = _config("local")
    attn = DRMAttention(config)
    metric_net = MetricNet(config.d_manifold, config.metric_rank, config.metric_hidden)
    x = torch.randn(2, 5, config.d_model)

    out = attn(x, metric_net)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert "dist_lr_fraction" in attn.last_distance_diagnostics
    assert torch.isfinite(attn.last_distance_diagnostics["dist_lr_fraction"])


def test_quadrature_mode_runs_and_is_finite():
    config = _config("quadrature", quad_points=3)
    attn = DRMAttention(config)
    metric_net = MetricNet(config.d_manifold, config.metric_rank, config.metric_hidden)
    x = torch.randn(1, 5, config.d_model)

    out = attn(x, metric_net)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert attn.last_attention is not None
    assert attn.last_attention.shape == (1, config.n_heads, 5, 5)
    for value in attn.last_distance_diagnostics.values():
        assert torch.isfinite(value)


def test_legacy_n_quad_enables_quadrature_when_requested():
    config = DRMTransformerConfig(
        d_model=16,
        n_heads=2,
        max_seq_len=8,
        d_manifold=4,
        metric_hidden=8,
        metric_rank=2,
        distance_mode="quadrature",
        n_quad=2,
    )
    attn = DRMAttention(config)

    assert attn.distance_mode == "quadrature"
    assert attn.quad_points == 2
