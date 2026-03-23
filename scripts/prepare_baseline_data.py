"""Prepara dataset de referencia fixo para baseline reproduzivel.

Gera um subset deterministico de Wikipedia EN (publico, sem auth) com
hash SHA256 verificavel. Qualquer pessoa pode reproduzir exatamente
o mesmo dataset.

Uso:
    python scripts/prepare_baseline_data.py
    python scripts/prepare_baseline_data.py --verify

O dataset e salvo em data/baseline/ com:
  - shards .npy (uint16 remapeado)
  - metadata.json (hash, config)
  - vocab_mapping.json
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Parametros fixos (NAO alterar — define o baseline canonico)
BASELINE_CONFIG = {
    "source": "wikipedia",
    "dataset": "wikimedia/wikipedia",
    "dataset_config": "20231101.en",
    "split": "train",
    "lang": "en",
    "max_tokens": 10_000_000,
    "vocab_size": 50000,
    "shard_size": 5_000_000,
    "min_text_len": 50,
    "max_articles": 50_000,
    "tokenizer": "o200k_base",
    "seed": 42,
    "val_fraction": 0.1,
}

OUTPUT_DIR = Path("data/baseline")


def prepare_baseline() -> dict:
    """Gera dataset baseline deterministico."""
    from datasets import load_dataset

    cfg = BASELINE_CONFIG
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[BASELINE] Gerando dataset de referencia fixo")
    logger.info("  source: %s (%s)", cfg["dataset"], cfg["dataset_config"])
    logger.info("  max_tokens: %dM", cfg["max_tokens"] // 1_000_000)
    logger.info("  vocab_size: %d", cfg["vocab_size"])

    # Stream Wikipedia EN
    ds = load_dataset(
        cfg["dataset"], cfg["dataset_config"],
        split=cfg["split"], streaming=True,
    )

    encoder = tiktoken.get_encoding(cfg["tokenizer"])
    freq = Counter()
    all_tokens = []
    total_tok = 0
    n_articles = 0
    chars_target = cfg["max_tokens"] * 4  # ~4 chars/token

    char_count = 0
    for example in tqdm(ds, desc="Streaming Wikipedia EN"):
        text = example.get("text", "")
        if len(text) < cfg["min_text_len"]:
            continue

        tokens = encoder.encode(text, disallowed_special=())
        freq.update(tokens)
        all_tokens.append(tokens)
        total_tok += len(tokens)
        char_count += len(text)
        n_articles += 1

        if char_count >= chars_target or n_articles >= cfg["max_articles"]:
            break

    logger.info("  %d artigos, %dM tokens", n_articles, total_tok // 1_000_000)

    # Vocab mapping top-K
    top_tokens = freq.most_common(cfg["vocab_size"] - 1)
    mapping = {}
    for new_id, (orig_id, _) in enumerate(top_tokens, start=1):
        mapping[orig_id] = new_id

    coverage = sum(c for _, c in top_tokens) / sum(freq.values()) * 100
    logger.info("  vocab coverage: %.1f%%", coverage)

    # Remapear todos os tokens
    all_remapped = []
    for tokens in all_tokens:
        all_remapped.extend(mapping.get(t, 0) for t in tokens)

    # Split train/val deterministico
    val_frac = cfg["val_fraction"]
    split_idx = int(len(all_remapped) * (1.0 - val_frac))
    train_tokens = all_remapped[:split_idx]
    val_tokens = all_remapped[split_idx:]

    logger.info("  train: %dM tokens, val: %dM tokens",
                len(train_tokens) // 1_000_000, len(val_tokens) // 1_000_000)

    # Salvar shards para train e val
    shard_hashes = []
    for split_name, split_tokens in [("train", train_tokens), ("val", val_tokens)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        buffer = list(split_tokens)
        shard_idx = 0
        shard_size = cfg["shard_size"]

        while len(buffer) >= shard_size:
            shard = np.array(buffer[:shard_size], dtype=np.uint16)
            path = split_dir / f"shard_{shard_idx:05d}.npy"
            np.save(path, shard)
            shard_hash = hashlib.sha256(shard.tobytes()).hexdigest()
            shard_hashes.append({"file": f"{split_name}/{path.name}", "sha256": shard_hash})
            shard_idx += 1
            buffer = buffer[shard_size:]

        if buffer:
            shard = np.array(buffer, dtype=np.uint16)
            path = split_dir / f"shard_{shard_idx:05d}.npy"
            np.save(path, shard)
            shard_hash = hashlib.sha256(shard.tobytes()).hexdigest()
            shard_hashes.append({"file": f"{split_name}/{path.name}", "sha256": shard_hash})
            shard_idx += 1

    total_saved = len(train_tokens) + len(val_tokens)

    # Hash global (hash de todos os hashes)
    global_hash = hashlib.sha256(
        "".join(h["sha256"] for h in shard_hashes).encode()
    ).hexdigest()

    # Metadata
    metadata = {
        "baseline_config": cfg,
        "total_tokens": total_saved,
        "n_articles": n_articles,
        "n_shards": shard_idx,
        "vocab_size": cfg["vocab_size"],
        "coverage_pct": round(coverage, 2),
        "shard_hashes": shard_hashes,
        "global_hash": global_hash,
        "remapped_vocab_size": cfg["vocab_size"],
        "tokenizer": cfg["tokenizer"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "vocab_mapping.json", "w") as f:
        json.dump(
            {"mapping": {str(k): v for k, v in mapping.items()},
             "metadata": metadata},
            f, indent=2,
        )

    logger.info("[BASELINE] Salvo em %s/", output_dir)
    logger.info("  %dM tokens, %d shards", total_saved // 1_000_000, shard_idx)
    logger.info("  global_hash: %s", global_hash)

    return metadata


def verify_baseline() -> bool:
    """Verifica integridade do dataset baseline via SHA256."""
    meta_path = OUTPUT_DIR / "metadata.json"
    if not meta_path.exists():
        logger.error("[VERIFY] metadata.json nao encontrado em %s", OUTPUT_DIR)
        return False

    with open(meta_path) as f:
        metadata = json.load(f)

    ok = True
    for entry in metadata["shard_hashes"]:
        path = OUTPUT_DIR / entry["file"]
        if not path.exists():
            logger.error("  FAIL: %s nao existe", entry["file"])
            ok = False
            continue
        shard = np.load(path)
        actual = hashlib.sha256(shard.tobytes()).hexdigest()
        if actual != entry["sha256"]:
            logger.error("  FAIL: %s hash mismatch", entry["file"])
            ok = False
        else:
            logger.info("  OK: %s", entry["file"])

    # Verificar hash global
    actual_global = hashlib.sha256(
        "".join(h["sha256"] for h in metadata["shard_hashes"]).encode()
    ).hexdigest()
    if actual_global != metadata["global_hash"]:
        logger.error("  FAIL: global hash mismatch")
        ok = False

    if ok:
        logger.info("[VERIFY] Dataset baseline integro (hash: %s)", metadata["global_hash"][:16])
    else:
        logger.error("[VERIFY] Dataset corrompido!")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Dataset baseline reproduzivel")
    parser.add_argument("--verify", action="store_true", help="Apenas verificar integridade")
    args = parser.parse_args()

    if args.verify:
        ok = verify_baseline()
        sys.exit(0 if ok else 1)
    else:
        prepare_baseline()


if __name__ == "__main__":
    main()
