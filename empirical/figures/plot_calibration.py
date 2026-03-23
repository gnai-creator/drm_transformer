"""Calibration dashboard: reliability diagram, confidence histogram, metricas."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import utils as _u
from utils import (
    set_seed, create_model, tokenize_texts, get_all_texts_and_labels,
    logger,
)


def _compute_all(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """Computa ECE, MCE, Brier, PPL e dados por bin."""
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_accs, bin_confs, bin_counts = [], [], []
    mce = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = mask.sum().item()
        bin_counts.append(count)
        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2)
        else:
            acc = accuracies[mask].mean().item()
            conf = confidences[mask].mean().item()
            bin_accs.append(acc)
            bin_confs.append(conf)
            mce = max(mce, abs(acc - conf))

    total = max(sum(bin_counts), 1)
    ece = sum(abs(a - c) * n / total for a, c, n in zip(bin_accs, bin_confs, bin_counts))

    # Brier
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    brier = float(((probs - one_hot) ** 2).sum(dim=-1).mean())

    # Perplexity
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(log_probs, targets, reduction="mean")
    ppl = float(torch.exp(nll))

    return {
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "confidences": confidences.cpu().numpy(),
        "ece": float(ece),
        "mce": float(mce),
        "brier": brier,
        "perplexity": ppl,
        "accuracy": float(accuracies.mean()),
        "mean_confidence": float(confidences.mean()),
    }


@torch.no_grad()
def main():
    set_seed(42)
    model, config = create_model()
    device = next(model.parameters()).device

    texts, _ = get_all_texts_and_labels()
    input_ids = tokenize_texts(texts, vocab_size=config.vocab_size).to(device)

    src = input_ids[:, :-1]
    tgt = input_ids[:, 1:]

    logits, _ = model(src)
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = tgt.reshape(B * T)

    mask = targets_flat > 0
    logits_flat = logits_flat[mask]
    targets_flat = targets_flat[mask]

    data = _compute_all(logits_flat, targets_flat)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- (0,0) Reliability Diagram ---
    ax = axes[0, 0]
    n_bins = len(data["bin_accs"])
    bin_width = 1.0 / n_bins
    positions = np.arange(n_bins) * bin_width + bin_width / 2

    gaps = data["bin_accs"] - data["bin_confs"]
    ax.bar(
        positions, data["bin_accs"], width=bin_width * 0.8,
        color="steelblue", edgecolor="black", linewidth=0.5,
        label="Accuracy",
    )
    ax.bar(
        positions, gaps, bottom=data["bin_confs"], width=bin_width * 0.8,
        color="coral", alpha=0.5, edgecolor="black", linewidth=0.5,
        label="Gap",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE={data['ece']:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Confidence Histogram ---
    ax = axes[0, 1]
    ax.hist(
        data["confidences"], bins=50, color="steelblue",
        edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    ax.axvline(
        data["mean_confidence"], color="red", linestyle="--",
        linewidth=1.5, label=f"Mean={data['mean_confidence']:.4f}",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Metricas de Calibracao (ECE, MCE, Brier) ---
    ax = axes[1, 0]
    metrics = ["ECE", "MCE", "Brier"]
    values = [data["ece"], data["mce"], data["brier"]]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax.set_ylabel("Score (lower = better)")
    ax.set_title("Calibration Metrics")
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")

    # --- (1,1) Perplexity + Accuracy ---
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = (
        f"Perplexity:  {data['perplexity']:.2f}\n"
        f"Accuracy:    {data['accuracy']:.4f}\n"
        f"Confidence:  {data['mean_confidence']:.4f}\n"
        f"\n"
        f"ECE:         {data['ece']:.4f}\n"
        f"MCE:         {data['mce']:.4f}\n"
        f"Brier:       {data['brier']:.4f}"
    )
    ax.text(
        0.5, 0.5, summary_text,
        transform=ax.transAxes, fontsize=16, fontfamily="monospace",
        verticalalignment="center", horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", edgecolor="gray"),
    )
    ax.set_title("Summary", fontsize=13)

    plt.suptitle("Calibration Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()

    _u.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _u.FIGURES_DIR / "calibration.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Salvo: %s", out_path)


if __name__ == "__main__":
    main()
