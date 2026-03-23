"""Utilitarios de reprodutibilidade para treinamento DRM Transformer.

Inclui:
- set_seed: fixa seeds globais (Python, NumPy, Torch, CUDA)
- set_deterministic: ativa flags deterministicas no backend
- build_run_manifest: gera JSON com info do ambiente, config, git, hardware
"""

import os
import json
import random
import hashlib
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Fixa seeds globais para reprodutibilidade.

    Args:
        seed: Seed inteiro.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("[REPRO] Seed=%d", seed)


def set_deterministic(warn_only: bool = True) -> None:
    """Ativa flags deterministicas no PyTorch.

    Args:
        warn_only: Se True, nao levanta erro quando op nao-deterministica
                   e usada (apenas warning). Default True para compatibilidade.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    logger.info("[REPRO] Deterministic mode ON (warn_only=%s)", warn_only)


def _git_info() -> Dict[str, str]:
    """Coleta info do git (commit hash, branch, dirty)."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["dirty"] = str(bool(status))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _hardware_info() -> Dict[str, Any]:
    """Coleta info de hardware."""
    hw = {
        "hostname": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 1)
        if hasattr(os, "sysconf") else "unknown",
    }

    if torch.cuda.is_available():
        hw["gpu_count"] = torch.cuda.device_count()
        hw["gpu_name"] = torch.cuda.get_device_name(0)
        hw["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1
        )
    else:
        hw["gpu_count"] = 0

    return hw


def _dependency_versions() -> Dict[str, str]:
    """Versoes das dependencias principais."""
    deps = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    cuda_ver = getattr(torch.version, "cuda", None)
    if cuda_ver:
        deps["cuda"] = cuda_ver

    cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    if cudnn_ver:
        deps["cudnn"] = str(cudnn_ver)

    try:
        import tiktoken
        deps["tiktoken"] = tiktoken.__version__
    except ImportError:
        pass

    return deps


def build_run_manifest(
    config: dict,
    seed: int,
    config_path: str = "",
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Constroi e salva manifest JSON do run.

    Args:
        config: Config final resolvida do treino.
        seed: Seed usada.
        config_path: Caminho do YAML de config.
        save_dir: Diretorio para salvar manifest (None = nao salva).

    Returns:
        Dict com manifest completo.
    """
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "config_path": config_path,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "git": _git_info(),
        "hardware": _hardware_info(),
        "dependencies": _dependency_versions(),
    }

    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "run_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("[REPRO] Manifest salvo: %s", manifest_path)

    return manifest
