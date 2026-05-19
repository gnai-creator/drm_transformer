"""Utilitarios para derivar subsets de shards tokenizados."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_shard(path: Path):
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".bin":
        return np.memmap(path, dtype=np.uint16, mode="r")
    raise ValueError(f"Formato de shard nao suportado: {path}")


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    if target.exists():
        target.unlink()

    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"Modo invalido: {mode}")


def _copy_json_if_exists(source_dir: Path, output_dir: Path, name: str) -> dict:
    source_path = source_dir / name
    if not source_path.exists():
        return {}

    with open(source_path, encoding="utf-8") as f:
        payload = json.load(f)

    with open(output_dir / name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def _write_subset_metadata(
    source_dir: Path,
    output_dir: Path,
    total_tokens: int,
    shard_count: int,
    max_tokens: int,
    mode: str,
) -> dict:
    source_metadata = _copy_json_if_exists(source_dir, output_dir, "metadata.json")
    _copy_json_if_exists(source_dir, output_dir, "vocab_mapping.json")

    metadata = dict(source_metadata)
    metadata.update({
        "total_tokens": total_tokens,
        "derived_from": str(source_dir),
        "source_total_tokens": source_metadata.get("total_tokens"),
        "subset_max_tokens": max_tokens,
        "subset_shards": shard_count,
        "subset_copy_mode": mode,
        "subset_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def derive_shard_subset(
    source_dir: Path,
    output_dir: Path,
    max_tokens: int,
    mode: str = "copy",
) -> dict:
    """Cria um dataset menor a partir dos primeiros tokens de outro dataset."""
    if max_tokens <= 0:
        raise ValueError("--max-tokens precisa ser maior que zero para subset")

    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_shards = sorted(
        list(source_dir.glob("shard_*.npy"))
        + list(source_dir.glob("shard_*.bin"))
    )
    if not source_shards:
        raise FileNotFoundError(f"Nenhum shard encontrado em {source_dir}")

    total = 0
    out_idx = 0
    for source_path in source_shards:
        shard = _load_shard(source_path)
        remaining = max_tokens - total
        if remaining <= 0:
            break

        if len(shard) <= remaining:
            target = output_dir / f"shard_{out_idx:05d}{source_path.suffix}"
            _link_or_copy(source_path, target, mode)
            total += len(shard)
            out_idx += 1
            continue

        target = output_dir / f"shard_{out_idx:05d}.npy"
        np.save(target, np.asarray(shard[:remaining]))
        total += remaining
        out_idx += 1
        break

    metadata = _write_subset_metadata(
        source_dir=source_dir,
        output_dir=output_dir,
        total_tokens=total,
        shard_count=out_idx,
        max_tokens=max_tokens,
        mode=mode,
    )

    logger.info(
        "[SUBSET DONE] %dM tokens em %d shards -> %s",
        total // 1_000_000,
        out_idx,
        output_dir,
    )
    return metadata
