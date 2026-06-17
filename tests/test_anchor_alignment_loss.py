import torch

from drm_transformer.losses import anchor_alignment_loss
from drm_transformer.metric_net import MetricNet


def test_anchor_alignment_loss_is_finite_and_backprops_to_metric_net():
    metric_net = MetricNet(dim=4, rank=2, hidden=8)
    coords = torch.rand(2, 5, 4)
    anchors = torch.rand(3, 4)

    U = metric_net(coords.detach().reshape(-1, 4)).view(2, 5, 4, 2)
    loss = anchor_alignment_loss(U, coords, anchors)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    grad = metric_net.net[-1].weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0


def test_anchor_alignment_is_integrated_in_trainer_drm_losses(tmp_path):
    from torch.utils.data import DataLoader

    from drm_transformer.config import DRMTransformerConfig
    from drm_transformer.model import DRMTransformer
    from drm_transformer.training.trainer import DRMTrainer

    model_config = DRMTransformerConfig(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
        d_manifold=4,
        metric_hidden=8,
        metric_rank=2,
        gravity_enabled=False,
        gamma_enabled=False,
        variable_dim=False,
    )
    model = DRMTransformer(model_config)
    loader = DataLoader([{"input_ids": torch.randint(0, 32, (8,)), "targets": torch.randint(0, 32, (8,))}])
    trainer = DRMTrainer(
        {
            "save_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
            "lambda_anchor_alignment": 1.0,
            "anchor_alignment_warmup_steps": 0,
            "geometry_warmup_steps": 0,
            "metric_diversity_warmup_steps": 10_000,
        },
        model,
        loader,
    )

    input_ids = torch.randint(0, 32, (2, 8))
    loss = trainer._compute_drm_losses(input_ids)
    loss.backward()

    assert torch.isfinite(loss)
    grad = model.metric_net.net[-1].weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0
