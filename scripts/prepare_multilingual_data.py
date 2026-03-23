"""Prepara dados multilingues para treino do DRM Transformer.

Pipeline streaming em 2 passes (memoria constante):
  Pass 1: Stream Wikipedia -> tokeniza -> conta freq -> salva shards raw
  Pass 2: Constroi vocab mapping -> remapeia shards raw -> salva shards finais

Uso:
    # Tudo de uma vez
    python scripts/prepare_multilingual_data.py \
        --output-dir data/multilingual \
        --max-tokens 20000000000 \
        --vocab-size 50000 \
        --langs en,pt,es,fr,de

    # Uma lingua por vez (resume automatico)
    python scripts/prepare_multilingual_data.py \
        --output-dir data/multilingual \
        --max-tokens 4000000000 \
        --vocab-size 50000 \
        --langs en

    python scripts/prepare_multilingual_data.py \
        --output-dir data/multilingual \
        --max-tokens 4000000000 \
        --vocab-size 50000 \
        --langs pt \
        --resume

    # Finalizar (remapeia tudo)
    python scripts/prepare_multilingual_data.py \
        --output-dir data/multilingual \
        --vocab-size 50000 \
        --finalize

Requisitos:
    pip install tiktoken datasets tqdm
"""

import argparse
import json
import logging
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

logger = logging.getLogger(__name__)

WIKI_LANGS = {
    "en": "20231101.en",
    "pt": "20231101.pt",
    "es": "20231101.es",
    "fr": "20231101.fr",
    "de": "20231101.de",
    "it": "20231101.it",
    "zh": "20231101.zh",
    "ja": "20231101.ja",
    "ru": "20231101.ru",
    "ar": "20231101.ar",
    "ko": "20231101.ko",
    "nl": "20231101.nl",
    "pl": "20231101.pl",
    "sv": "20231101.sv",
    "tr": "20231101.tr",
}

SHARD_SIZE = 5_000_000


def _raw_dir(output_dir: Path) -> Path:
    return output_dir / "_raw"


def _freq_path(output_dir: Path) -> Path:
    return output_dir / "_raw" / "freq.json"


def _state_path(output_dir: Path) -> Path:
    return output_dir / "_raw" / "state.json"


def _load_state(output_dir: Path) -> dict:
    sp = _state_path(output_dir)
    if sp.exists():
        with open(sp) as f:
            return json.load(f)
    return {"completed_langs": [], "total_tokens_raw": 0, "shard_idx": 0}


def _save_state(output_dir: Path, state: dict):
    sp = _state_path(output_dir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(state, f, indent=2)


def _load_freq(output_dir: Path) -> Counter:
    fp = _freq_path(output_dir)
    if fp.exists():
        with open(fp) as f:
            data = json.load(f)
        return Counter({int(k): v for k, v in data.items()})
    return Counter()


def _save_freq(output_dir: Path, freq: Counter):
    fp = _freq_path(output_dir)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        json.dump({str(k): v for k, v in freq.items()}, f)


def pass1_stream_and_save_raw(
    langs: list,
    max_tokens: int,
    output_dir: Path,
    shard_size: int,
    resume: bool,
):
    """Pass 1: stream textos, tokeniza, salva shards raw uint32, conta freq."""
    from datasets import load_dataset

    raw_dir = _raw_dir(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(output_dir) if resume else {
        "completed_langs": [], "total_tokens_raw": 0, "shard_idx": 0,
    }
    freq = _load_freq(output_dir) if resume else Counter()

    encoder = tiktoken.get_encoding("o200k_base")
    tokens_per_lang = max_tokens // len(langs) if max_tokens > 0 else 0

    buffer = []
    shard_idx = state["shard_idx"]
    total_raw = state["total_tokens_raw"]

    for lang in langs:
        if lang in state["completed_langs"]:
            logger.info("[SKIP] %s (ja processado)", lang)
            continue

        logger.info("[PASS1] %s: ~%dM tokens alvo", lang,
                    (tokens_per_lang // 1_000_000) if tokens_per_lang else 0)

        wiki_config = WIKI_LANGS.get(lang, f"20231101.{lang}")
        try:
            ds = load_dataset(
                "wikimedia/wikipedia", wiki_config,
                split="train", streaming=True,
            )
        except Exception as e:
            logger.warning("[WARN] %s nao disponivel: %s", lang, e)
            continue

        lang_tokens = 0
        chars_target = tokens_per_lang * 4 if tokens_per_lang else float("inf")
        char_count = 0

        for example in tqdm(ds, desc=f"Streaming {lang}"):
            text = example.get("text", "")
            if len(text) < 50:
                continue

            tokens = encoder.encode(text, allowed_special=set())
            freq.update(tokens)
            buffer.extend(tokens)
            lang_tokens += len(tokens)
            char_count += len(text)

            # Flush buffer para disco quando cheio
            while len(buffer) >= shard_size:
                shard = np.array(buffer[:shard_size], dtype=np.uint32)
                path = raw_dir / f"raw_{shard_idx:05d}.npy"
                np.save(path, shard)
                total_raw += shard_size
                shard_idx += 1
                buffer = buffer[shard_size:]

            if char_count >= chars_target:
                break

        logger.info("  %s: %dM tokens", lang, lang_tokens // 1_000_000)

        state["completed_langs"].append(lang)
        state["total_tokens_raw"] = total_raw + len(buffer)
        state["shard_idx"] = shard_idx
        _save_state(output_dir, state)
        _save_freq(output_dir, freq)
        logger.info("  [CHECKPOINT] estado salvo (%d langs, %dM tokens)",
                    len(state["completed_langs"]),
                    state["total_tokens_raw"] // 1_000_000)

    # Shard final parcial
    if buffer:
        shard = np.array(buffer, dtype=np.uint32)
        path = raw_dir / f"raw_{shard_idx:05d}.npy"
        np.save(path, shard)
        total_raw += len(buffer)
        shard_idx += 1

    state["total_tokens_raw"] = total_raw
    state["shard_idx"] = shard_idx
    _save_state(output_dir, state)
    _save_freq(output_dir, freq)

    logger.info("[PASS1 DONE] %dM tokens raw em %d shards",
                total_raw // 1_000_000, shard_idx)
    return freq


def build_vocab_mapping(freq: Counter, target_vocab_size: int) -> dict:
    """Cria mapping dos top-K tokens para IDs compactos (0 = UNK)."""
    top_tokens = freq.most_common(target_vocab_size - 1)

    mapping = {}
    for new_id, (orig_id, _) in enumerate(top_tokens, start=1):
        mapping[orig_id] = new_id

    coverage = sum(c for _, c in top_tokens)
    total = sum(freq.values())

    logger.info("[VOCAB] %d -> %d tokens (cobertura: %.2f%%)",
                len(freq), target_vocab_size, 100.0 * coverage / total)

    return mapping


def pass2_remap_shards(
    output_dir: Path,
    mapping: dict,
    max_tokens: int,
):
    """Pass 2: le shards raw, remapeia com vocab mapping, salva shards uint16."""
    raw_dir = _raw_dir(output_dir)
    raw_files = sorted(raw_dir.glob("raw_*.npy"))

    if not raw_files:
        logger.error("[ERROR] Nenhum shard raw encontrado em %s", raw_dir)
        return 0

    total_saved = 0
    out_shard_idx = 0
    buffer = np.array([], dtype=np.uint16)

    for raw_path in tqdm(raw_files, desc="Remapping shards"):
        raw = np.load(raw_path)
        # Vectorized remap: lookup com default 0 (UNK)
        remapped = np.zeros(len(raw), dtype=np.uint16)
        for orig_id, new_id in mapping.items():
            remapped[raw == orig_id] = new_id

        buffer = np.concatenate([buffer, remapped])

        while len(buffer) >= SHARD_SIZE:
            shard = buffer[:SHARD_SIZE]
            path = output_dir / f"shard_{out_shard_idx:05d}.npy"
            np.save(path, shard)
            total_saved += SHARD_SIZE
            out_shard_idx += 1
            buffer = buffer[SHARD_SIZE:]

            if 0 < max_tokens <= total_saved:
                break

        if 0 < max_tokens <= total_saved:
            break

    # Shard final
    if len(buffer) > 0 and (max_tokens <= 0 or total_saved < max_tokens):
        path = output_dir / f"shard_{out_shard_idx:05d}.npy"
        np.save(path, buffer)
        total_saved += len(buffer)
        out_shard_idx += 1

    logger.info("[PASS2 DONE] %dM tokens em %d shards -> %s/",
                total_saved // 1_000_000, out_shard_idx, output_dir)

    return total_saved


def finalize(output_dir: Path, vocab_size: int, max_tokens: int):
    """Constroi mapping e remapeia todos os shards raw."""
    freq = _load_freq(output_dir)
    if not freq:
        logger.error("[ERROR] Nenhuma frequencia encontrada. Rode pass1 primeiro.")
        return

    state = _load_state(output_dir)
    logger.info("[FINALIZE] %dM tokens raw, %d langs: %s",
                state["total_tokens_raw"] // 1_000_000,
                len(state["completed_langs"]),
                ", ".join(state["completed_langs"]))

    mapping = build_vocab_mapping(freq, vocab_size)
    total = pass2_remap_shards(output_dir, mapping, max_tokens)

    # Metadata
    metadata = {
        "tokenizer": "o200k_base",
        "original_vocab_size": tiktoken.get_encoding("o200k_base").n_vocab,
        "remapped_vocab_size": vocab_size,
        "total_tokens": total,
        "langs": state["completed_langs"],
        "coverage_pct": sum(
            freq[t] for t in mapping
        ) / sum(freq.values()) * 100,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    mapping_path = output_dir / "vocab_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(
            {"mapping": {str(k): v for k, v in mapping.items()},
             "metadata": metadata},
            f, indent=2,
        )

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[DONE] %dM tokens, vocab=%d, cobertura=%.1f%%",
                total // 1_000_000, vocab_size, metadata["coverage_pct"])

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepara dados multilingues para DRM Transformer"
    )
    parser.add_argument(
        "--output-dir", default="data/multilingual",
        help="Diretorio de saida para shards",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=0,
        help="Total maximo de tokens (0 = sem limite)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=50000,
        help="Tamanho do vocabulario remapeado",
    )
    parser.add_argument(
        "--langs", default="en,pt,es,fr,de",
        help="Linguas separadas por virgula",
    )
    parser.add_argument(
        "--shard-size", type=int, default=SHARD_SIZE,
        help="Tokens por shard",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Retomar de checkpoint anterior",
    )
    parser.add_argument(
        "--finalize", action="store_true",
        help="Apenas construir mapping e remapear shards (skip download)",
    )
    parser.add_argument(
        "--clean-raw", action="store_true",
        help="Apagar shards raw apos finalizacao",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        metadata = finalize(output_dir, args.vocab_size, args.max_tokens)
    else:
        langs = [lang.strip() for lang in args.langs.split(",")]

        # Pass 1: stream e salva raw
        freq = pass1_stream_and_save_raw(
            langs, args.max_tokens, output_dir,
            args.shard_size, args.resume,
        )

        # Pass 2: remap
        finalize(output_dir, args.vocab_size, args.max_tokens)

    if args.clean_raw:
        raw_dir = _raw_dir(output_dir)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            logger.info("[CLEAN] Shards raw removidos")

    elapsed = time.time() - t0
    logger.info("[TEMPO] %.0fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
