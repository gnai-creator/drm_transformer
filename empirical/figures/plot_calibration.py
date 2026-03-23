"""Reliability diagram + confidence histogram para calibracao."""

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


def _bin_calibration(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """Computa dados por bin para reliability diagram."""
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = mask.sum().item()
        bin_counts.append(count)
        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2)
        else:
            bin_accs.append(accuracies[mask].mean().item())
            bin_confs.append(confidences[mask].mean().item())

    return {
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "confidences": confidences.cpu().numpy(),
        "ece": float(sum(
            abs(a - c) * n / max(sum(bin_counts), 1)
            for a, c, n in zip(bin_accs, bin_confs, bin_counts)
        )),
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

    data = _bin_calibration(logits_flat, targets_flat)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Reliability Diagram ---
    n_bins = len(data["bin_accs"])
    bin_width = 1.0 / n_bins
    positions = np.arange(n_bins) * bin_width + bin_width / 2

    # Barras de gap (erro)
    gaps = data["bin_accs"] - data["bin_confs"]
    ax1.bar(
        positions, data["bin_accs"], width=bin_width * 0.8,
        color="steelblue", edgecolor="black", linewidth=0.5,
        label="Accuracy",
    )
    ax1.bar(
        positions, gaps, bottom=data["bin_confs"], width=bin_width * 0.8,
        color="coral", alpha=0.5, edgecolor="black", linewidth=0.5,
        label="Gap",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"Reliability Diagram (ECE={data['ece']:.4f})", fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Confidence Histogram ---
    ax2.hist(
        data["confidences"], bins=50, color="steelblue",
        edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    ax2.axvline(
        data["confidences"].mean(), color="red", linestyle="--",
        linewidth=1.5, label=f"Mean={data['confidences'].mean():.4f}",
    )
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Distribution", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Calibration Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    _u.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _u.FIGURES_DIR / "calibration.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Salvo: %s", out_path)


if __name__ == "__main__":
    main()
