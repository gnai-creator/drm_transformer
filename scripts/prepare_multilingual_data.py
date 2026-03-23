"""Prepara dados multilingues para treino do DRM Transformer.

Pipeline:
1. Baixa Wikipedia multilingue do HuggingFace (publico, sem auth)
2. Tokeniza com tiktoken o200k_base
3. Conta frequencias e cria mapping para top-K tokens
4. Remapeia tokens e salva shards uint16

Uso:
    python scripts/prepare_multilingual_data.py \
        --output-dir data/multilingual \
        --max-tokens 50000000 \
        --vocab-size 50000 \
        --langs en,pt,es,fr,de

Requisitos:
    pip install tiktoken datasets tqdm
"""

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Wikipedia multilingual (publico, sem auth)
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


def load_texts(
    langs: list,
    max_tokens: int,
    tokens_per_lang: int = 0,
) -> list:
    """Carrega textos do FineWeb-Edu 2 via streaming.

    Args:
        langs: Lista de codigos de lingua (ex: ["en", "pt"]).
        max_tokens: Total maximo de tokens (estimado ~4 chars/token).
        tokens_per_lang: Tokens por lingua (0 = dividir igualmente).

    Returns:
        Lista de strings de texto.
    """
    from datasets import load_dataset

    if tokens_per_lang == 0:
        tokens_per_lang = max_tokens // len(langs)

    # ~4 chars por token em media
    chars_per_lang = tokens_per_lang * 4
    texts = []

    for lang in langs:
        logger.info("[LOAD] %s: ~%dK tokens", lang, tokens_per_lang // 1000)

        wiki_config = WIKI_LANGS.get(lang, f"20231101.{lang}")

        try:
            ds = load_dataset(
                "wikimedia/wikipedia",
                wiki_config,
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.warning("[WARN] %s nao disponivel: %s", lang, e)
            continue

        char_count = 0
        lang_start = len(texts)
        for example in ds:
            text = example.get("text", "")
            if len(text) < 50:
                continue
            texts.append(text)
            char_count += len(text)
            if char_count >= chars_per_lang:
                break

        n_texts = len(texts) - lang_start
        logger.info("  %s: %d textos, ~%dK chars", lang, n_texts, char_count // 1000)

    return texts


def tokenize_and_count(
    texts: list,
    encoder: tiktoken.Encoding,
) -> tuple:
    """Tokeniza textos e conta frequencias.

    Args:
        texts: Lista de strings.
        encoder: Encoder tiktoken.

    Returns:
        (all_tokens: list[list[int]], freq: Counter)
    """
    logger.info("[TOKENIZE] %d textos com %s", len(texts), encoder.name)

    freq = Counter()
    all_tokens = []

    for text in tqdm(texts, desc="Tokenizing"):
        tokens = encoder.encode(text, allowed_special=set())
        all_tokens.append(tokens)
        freq.update(tokens)

    total = sum(freq.values())
    logger.info(
        "  %d tokens totais, %d tipos unicos",
        total, len(freq),
    )

    return all_tokens, freq


def build_vocab_mapping(
    freq: Counter,
    target_vocab_size: int,
) -> dict:
    """Cria mapping dos top-K tokens para IDs compactos.

    Token 0 reservado para UNK. Tokens fora do top-K mapeados para 0.

    Args:
        freq: Frequencias de tokens.
        target_vocab_size: Tamanho alvo do vocabulario.

    Returns:
        Dict {original_id: new_id}, onde new_id in [0, target_vocab_size).
    """
    # Top K-1 tokens (reservar 0 para UNK)
    top_tokens = freq.most_common(target_vocab_size - 1)

    mapping = {}
    for new_id, (orig_id, count) in enumerate(top_tokens, start=1):
        mapping[orig_id] = new_id

    coverage = sum(c for _, c in top_tokens)
    total = sum(freq.values())

    logger.info(
        "[VOCAB] %d -> %d tokens (cobertura: %.2f%%)",
        len(freq), target_vocab_size,
        100.0 * coverage / total,
    )

    return mapping


def remap_and_save(
    all_tokens: list,
    mapping: dict,
    output_dir: Path,
    shard_size: int = 5_000_000,
    max_tokens: int = 0,
) -> int:
    """Remapeia tokens e salva shards uint16.

    Args:
        all_tokens: Lista de listas de token IDs originais.
        mapping: Dict {original_id: new_id}.
        output_dir: Diretorio de saida.
        shard_size: Tokens por shard.
        max_tokens: Limite total (0 = sem limite).

    Returns:
        Total de tokens salvos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer = []
    shard_idx = 0
    total_saved = 0

    for tokens in all_tokens:
        remapped = [mapping.get(t, 0) for t in tokens]
        buffer.extend(remapped)

        while len(buffer) >= shard_size:
            shard = np.array(buffer[:shard_size], dtype=np.uint16)
            path = output_dir / f"shard_{shard_idx:05d}.npy"
            np.save(path, shard)
            total_saved += len(shard)
            shard_idx += 1
            buffer = buffer[shard_size:]

            if max_tokens > 0 and total_saved >= max_tokens:
                break

        if max_tokens > 0 and total_saved >= max_tokens:
            break

    # Shard final
    if buffer and (max_tokens == 0 or total_saved < max_tokens):
        shard = np.array(buffer, dtype=np.uint16)
        path = output_dir / f"shard_{shard_idx:05d}.npy"
        np.save(path, shard)
        total_saved += len(shard)
        shard_idx += 1

    logger.info(
        "[SAVE] %d tokens em %d shards -> %s/",
        total_saved, shard_idx, output_dir,
    )

    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Prepara dados multilingues para DRM Transformer"
    )
    parser.add_argument(
        "--output-dir", default="data/multilingual",
        help="Diretorio de saida para shards",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50_000_000,
        help="Total maximo de tokens",
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
        "--shard-size", type=int, default=5_000_000,
        help="Tokens por shard",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    t0 = time.time()
    langs = [l.strip() for l in args.langs.split(",")]

    # 1. Carregar textos
    texts = load_texts(langs, args.max_tokens)
    if not texts:
        logger.error("[ERROR] Nenhum texto carregado")
        return

    # 2. Tokenizar com o200k_base
    encoder = tiktoken.get_encoding("o200k_base")
    all_tokens, freq = tokenize_and_count(texts, encoder)

    # 3. Construir mapping top-K
    mapping = build_vocab_mapping(freq, args.vocab_size)

    # 4. Remapear e salvar
    output_dir = Path(args.output_dir)
    total = remap_and_save(
        all_tokens, mapping, output_dir,
        shard_size=args.shard_size,
        max_tokens=args.max_tokens,
    )

    # 5. Salvar metadata
    metadata = {
        "tokenizer": "o200k_base",
        "original_vocab_size": encoder.n_vocab,
        "remapped_vocab_size": args.vocab_size,
        "total_tokens": total,
        "langs": langs,
        "coverage_pct": sum(
            freq[t] for t in mapping
        ) / sum(freq.values()) * 100,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Salvar mapping para decode posterior
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

    elapsed = time.time() - t0
    logger.info(
        "[DONE] %d tokens, vocab=%d, %d langs em %.0fs",
        total, args.vocab_size, len(langs), elapsed,
    )
    logger.info("  Cobertura: %.1f%%", metadata["coverage_pct"])


if __name__ == "__main__":
    main()
