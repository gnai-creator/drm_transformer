"""Setup distribuido: DDP, FSDP, mixed precision, gradient checkpointing."""

import os
import sys
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict

logger = logging.getLogger(__name__)


def setup_distributed(config: dict) -> Dict:
    """Inicializa ambiente distribuido.

    Detecta automaticamente se esta rodando via torchrun.

    Args:
        config: Dict de configuracao com flags distributed, dist_backend, etc.

    Returns:
        Dict com rank, local_rank, world_size, device, is_main.
    """
    if not config.get("distributed", False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": device,
            "is_main": True,
        }

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        backend = config.get("dist_backend", "nccl")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def cleanup_distributed() -> None:
    """Finaliza processo distribuido."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: torch.nn.Module,
    config: dict,
    device: str,
) -> torch.nn.Module:
    """Envolve modelo com DDP ou FSDP.

    Args:
        model: Modelo base.
        config: Dict de configuracao.
        device: Device string.

    Returns:
        Modelo wrapped.
    """
    model = model.to(device)

    if config.get("gradient_checkpointing", False):
        _enable_gradient_checkpointing(model)

    if config.get("compile_model", False) and hasattr(torch, "compile"):
        if sys.platform == "win32":
            logger.warning("[DIST] torch.compile nao suportado no Windows, ignorando")
        else:
            model = torch.compile(model)
            logger.info("[DIST] torch.compile ativado")

    if not config.get("distributed", False):
        return model

    if config.get("fsdp", False):
        return _wrap_fsdp(model, config, device)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    logger.info("[DIST] DDP ativado (local_rank=%d)", local_rank)
    return model


def _wrap_fsdp(model: torch.nn.Module, config: dict, device: str) -> torch.nn.Module:
    """Envolve modelo com FSDP."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

    mp = config.get("mixed_precision", "bf16")
    if mp == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif mp == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None

    sharding = config.get("fsdp_sharding", "full")
    strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad": ShardingStrategy.SHARD_GRAD_OP,
        "no": ShardingStrategy.NO_SHARD,
    }.get(sharding, ShardingStrategy.FULL_SHARD)

    model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        device_id=torch.device(device),
    )
    logger.info("[DIST] FSDP ativado (strategy=%s, mp=%s)", sharding, mp)
    return model


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Ativa gradient checkpointing nos blocos do transformer."""
    from torch.utils.checkpoint import checkpoint

    count = 0
    for module in model.modules():
        if hasattr(module, "forward") and module.__class__.__name__ == "DRMTransformerBlock":
            original_forward = module.forward

            def make_ckpt_forward(orig):
                def ckpt_forward(*args, **kwargs):
                    return checkpoint(orig, *args, use_reentrant=False, **kwargs)
                return ckpt_forward

            module.forward = make_ckpt_forward(original_forward)
            count += 1

    if count > 0:
        logger.info("[DIST] Gradient checkpointing ativado em %d blocos", count)
