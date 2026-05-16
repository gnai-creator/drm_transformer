"""Dataset e dataloader para treinamento com shards tokenizados."""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


class ShardedDataset(Dataset):
    """Dataset que le shards tokenizados (uint16/uint32 .npy ou uint16 .bin).

    Cada shard e mantido em mmap quando possivel. O dataset retorna janelas
    de tamanho seq_len + 1 sem concatenar todos os tokens em RAM.

    Args:
        data_dir: Diretorio com shards .npy ou .bin.
        seq_len: Comprimento da sequencia.
        max_tokens: Limite de tokens total (0 = sem limite).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 1024,
        max_tokens: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len

        data_path = Path(data_dir)
        shard_files = sorted(
            list(data_path.glob("*.npy")) + list(data_path.glob("*.bin"))
        )
        if not shard_files:
            raise FileNotFoundError(
                f"Nenhum shard encontrado em {data_dir}"
            )

        self.shards = []
        self.shard_lengths = []
        self.cum_lengths = []

        total = 0
        for sf in shard_files:
            if sf.suffix == ".npy":
                arr = np.load(str(sf), mmap_mode="r")
            else:
                arr = np.memmap(str(sf), dtype=np.uint16, mode="r")

            remaining = max_tokens - total if max_tokens > 0 else len(arr)
            length = min(len(arr), remaining)
            if length <= 0:
                break

            self.shards.append(arr)
            self.shard_lengths.append(length)
            total += length
            self.cum_lengths.append(total)

            if max_tokens > 0 and total >= max_tokens:
                break

        self.total_tokens = total
        self.n_samples = max((self.total_tokens - 1) // seq_len, 0)

        logger.info(
            "[DATA] %d tokens, %d samples (seq_len=%d) de %d shards",
            self.total_tokens, self.n_samples, seq_len, len(self.shards),
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(idx)

        start = idx * self.seq_len
        chunk = self._read_tokens(start, self.seq_len + 1)
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return {"input_ids": x, "targets": y}

    def _read_tokens(self, start: int, length: int) -> np.ndarray:
        """Le uma janela global, atravessando shards quando necessario."""
        shard_idx = int(np.searchsorted(self.cum_lengths, start, side="right"))
        prev_end = 0 if shard_idx == 0 else self.cum_lengths[shard_idx - 1]
        offset = start - prev_end

        shard_len = self.shard_lengths[shard_idx]
        if offset + length <= shard_len:
            return np.asarray(self.shards[shard_idx][offset:offset + length])

        parts = []
        remaining = length
        while remaining > 0:
            shard = self.shards[shard_idx]
            shard_len = self.shard_lengths[shard_idx]
            take = min(remaining, shard_len - offset)
            parts.append(np.asarray(shard[offset:offset + take]))
            remaining -= take
            shard_idx += 1
            offset = 0

        return np.concatenate(parts)

    def close(self) -> None:
        """Fecha handles de mmap mantidos pelos shards, quando existirem."""
        for shard in self.shards:
            mmap_obj = getattr(shard, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        self.shards = []
        self.shard_lengths = []
        self.cum_lengths = []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_dataloader(
    data_dir: str,
    seq_len: int = 1024,
    batch_size: int = 16,
    max_tokens: int = 0,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 2,
) -> DataLoader:
    """Cria dataloader com suporte distribuido.

    Args:
        data_dir: Diretorio de shards.
        seq_len: Comprimento da sequencia.
        batch_size: Batch size por GPU.
        max_tokens: Limite de tokens.
        rank: Rank do processo atual.
        world_size: Numero total de processos.
        num_workers: Workers do dataloader.

    Returns:
        DataLoader configurado.
    """
    dataset = ShardedDataset(data_dir, seq_len, max_tokens)

    sampler: Optional[DistributedSampler] = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
