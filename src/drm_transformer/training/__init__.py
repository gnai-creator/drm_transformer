"""Modulo de treinamento distribuido do DRM Transformer."""

from .distributed import setup_distributed, cleanup_distributed, wrap_model_ddp
from .trainer import DRMTrainer
from .data import ShardedDataset, create_dataloader
