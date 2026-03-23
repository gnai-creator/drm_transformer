"""Metricas de calibracao: ECE, MCE, Brier Score, Perplexity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import numpy as np
from utils import (
    set_seed, create_model, tokenize_texts, get_all_texts_and_labels,
    save_results, logger,
)


def _compute_calibration(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """Computa ECE, MCE, Brier Score e Perplexity.

    Args:
        logits: [N, V] logits do modelo.
        targets: [N] token IDs alvo.
        n_bins: Numero de bins para ECE/MCE.

    Returns:
        Dict com ece, mce, brier, perplexity.
    """
    probs = F.softmax(logits, dim=-1)  # [N, V]
    confidences, predictions = probs.max(dim=-1)  # [N], [N]
    accuracies = (predictions == targets).float()

    # --- ECE e MCE (bins adaptativos / equal-mass) ---
    conf_sorted, sort_idx = confidences.sort()
    acc_sorted = accuracies[sort_idx]
    chunk = max(len(conf_sorted) // n_bins, 1)

    ece = torch.tensor(0.0, device=logits.device)
    mce = torch.tensor(0.0, device=logits.device)
    total = len(conf_sorted)

    for i in range(n_bins):
        lo = i * chunk
        hi = min((i + 1) * chunk, total)
        if lo >= total:
            break
        bin_acc = acc_sorted[lo:hi].mean()
        bin_conf = conf_sorted[lo:hi].mean()
        bin_size = (hi - lo) / total
        gap = (bin_acc - bin_conf).abs()
        ece += gap * bin_size
        mce = torch.max(mce, gap)

    # --- Brier Score ---
    # Brier multiclasse normalizado: mean(sum((one_hot - probs)^2)) / 2
    # Range [0, 1]. Random baseline = 1 - 1/V.
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean() / 2.0

    # --- Perplexity ---
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(log_probs, targets, reduction="mean")
    perplexity = torch.exp(nll)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "perplexity": float(perplexity),
        "mean_confidence": float(confidences.mean()),
        "accuracy": float(accuracies.mean()),
    }


@torch.no_grad()
def run(seed: int = 42) -> dict:
    """Executa avaliacao de calibracao."""
    set_seed(seed)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, labels = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts, vocab_size=config.vocab_size).to(device)

    # Forward: input[:-1] -> target[1:]
    src = input_ids[:, :-1]
    tgt = input_ids[:, 1:]

    logits, _ = model(src)  # [B, T, V]

    # Flatten
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = tgt.reshape(B * T)

    # Filtrar padding (token 0)
    mask = targets_flat > 0
    logits_flat = logits_flat[mask]
    targets_flat = targets_flat[mask]

    metrics = _compute_calibration(logits_flat, targets_flat)

    logger.info("ECE:        %.4f", metrics["ece"])
    logger.info("MCE:        %.4f", metrics["mce"])
    logger.info("Brier:      %.4f", metrics["brier"])
    logger.info("Perplexity: %.2f", metrics["perplexity"])
    logger.info("Confidence: %.4f", metrics["mean_confidence"])
    logger.info("Accuracy:   %.4f", metrics["accuracy"])

    return metrics


def main():
    all_seeds = {}
    for seed in [42, 123, 7]:
        logger.info("\n=== Seed %d ===", seed)
        all_seeds[f"seed_{seed}"] = run(seed)

    # Media
    keys = ["ece", "mce", "brier", "perplexity", "mean_confidence", "accuracy"]
    summary = {}
    for k in keys:
        vals = [all_seeds[s][k] for s in all_seeds]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        summary[k] = {"mean": mean, "std": std}
        logger.info("[RESUMO] %s: %.4f +/- %.4f", k, mean, std)

    all_seeds["summary"] = summary
    save_results({"calibration": all_seeds})
    logger.info("Teste de calibracao concluido.")


if __name__ == "__main__":
    main()
