import torch

from drm_transformer.attention import DRMAttention
from drm_transformer.config import DRMTransformerConfig
from drm_transformer.metric_net import MetricNet


def test_metric_diagnostics_are_computed_and_bounded():
    config = DRMTransformerConfig(
        d_model=16,
        n_heads=2,
        max_seq_len=8,
        d_manifold=4,
        metric_hidden=8,
        metric_rank=2,
        dropout=0.0,
        gamma_enabled=False,
        gravity_enabled=False,
        variable_dim=False,
    )
    attn = DRMAttention(config)
    metric_net = MetricNet(config.d_manifold, config.metric_rank, config.metric_hidden)
    x = torch.randn(2, 6, config.d_model)

    _ = attn(x, metric_net)
    metrics = attn.last_distance_diagnostics

    expected = {
        "metric_U_norm_mean",
        "metric_U_norm_std",
        "metric_U_variance",
        "metric_condition_proxy",
        "geodesic_vs_euclidean_delta_mean",
        "geodesic_vs_euclidean_delta_std",
        "dist_lr_fraction",
    }
    assert expected.issubset(metrics)
    for key in expected:
        assert torch.isfinite(metrics[key]), key

    frac = metrics["dist_lr_fraction"].item()
    assert 0.0 <= frac <= 1.0


def test_dist_lr_fraction_detects_nonzero_metric_term():
    U = torch.ones(1, 1, 2, 3, 1)
    q = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]]]])
    k = torch.tensor([[[[0.25, 0.0, 0.0], [0.0, 0.5, 1.0]]]])

    dist_sq, dist_euc, dist_lr = DRMAttention._local_distance(None, q, k, U)
    metrics = DRMAttention._distance_diagnostics(U, dist_euc, dist_lr)

    assert torch.isfinite(dist_sq).all()
    assert metrics["dist_lr_fraction"].item() > 0.01
