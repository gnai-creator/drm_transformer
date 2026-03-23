"""Dataset e dataloader para treinamento com shards tokenizados."""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


class ShardedDataset(Dataset):
    """Dataset que le shards tokenizados (uint16 .npy/.bin).

    Cada shard e um array 1D de token IDs. O dataset concatena
    todos os shards e retorna janelas de tamanho seq_len + 1
    (input + target deslocado por 1).

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

        chunks = []
        total = 0
        for sf in shard_files:
            if sf.suffix == ".npy":
                arr = np.load(str(sf))
            else:
                arr = np.fromfile(str(sf), dtype=np.uint16)
            chunks.append(arr)
            total += len(arr)
            if max_tokens > 0 and total >= max_tokens:
                break

        self.data = np.concatenate(chunks).astype(np.int64)
        if max_tokens > 0:
            self.data = self.data[:max_tokens]

        self.n_samples = (len(self.data) - 1) // seq_len

        logger.info(
            "[DATA] %d tokens, %d samples (seq_len=%d) de %d shards",
            len(self.data), self.n_samples, seq_len, len(shard_files),
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return {"input_ids": x, "targets": y}


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
