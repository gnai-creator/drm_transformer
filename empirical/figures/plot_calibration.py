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
    """Computa ECE, MCE, Brier, PPL com bins adaptativos (equal-mass)."""
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).float()

    # Bins adaptativos: quantis da distribuicao de confidence
    conf_np = confidences.cpu().numpy()
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_boundaries = np.quantile(conf_np, quantiles)
    # Garantir extremos
    bin_boundaries[0] = 0.0
    bin_boundaries[-1] = 1.0 + 1e-8

    bin_accs, bin_confs, bin_counts = [], [], []
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum().item()
        bin_counts.append(count)
        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
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
        "bin_boundaries": bin_boundaries,
        "confidences": conf_np,
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

    # --- (0,0) Reliability Diagram (bins adaptativos) ---
    ax = axes[0, 0]
    n_bins = len(data["bin_accs"])
    boundaries = data["bin_boundaries"]

    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        width = hi - lo
        center = (lo + hi) / 2
        acc = data["bin_accs"][i]
        conf = data["bin_confs"][i]

        # Barra de accuracy
        ax.bar(center, acc, width=width * 0.9, color="steelblue",
               edgecolor="black", linewidth=0.5, alpha=0.8)
        # Barra de gap (|acc - conf|) em coral
        if data["bin_counts"][i] > 0:
            gap = abs(acc - conf)
            gap_bottom = min(acc, conf)
            ax.bar(center, gap, bottom=gap_bottom, width=width * 0.9,
                   color="coral", alpha=0.4, edgecolor="none")

    # Linha de calibracao perfeita no range real
    conf_range = [data["confidences"].min(), data["confidences"].max()]
    ax.plot(conf_range, conf_range, "k--", linewidth=1.5, label="Perfect", zorder=5)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — Adaptive Bins (ECE={data['ece']:.4f})")
    # Zoom no range real de confidence
    margin = (conf_range[1] - conf_range[0]) * 0.1
    ax.set_xlim(conf_range[0] - margin, conf_range[1] + margin)
    ax.set_ylim(0, max(max(data["bin_accs"]) * 1.2, 0.05))
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
        linewidth=1.5, label=f"Mean={data['mean_confidence']:.6f}",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Metricas separadas: ECE e MCE (eixo esquerdo), Brier (eixo direito) ---
    ax = axes[1, 0]
    x_pos = np.array([0, 1])
    bars_left = ax.bar(
        x_pos, [data["ece"], data["mce"]], width=0.6,
        color=["#4c72b0", "#dd8452"], edgecolor="black", linewidth=0.5,
        label="ECE / MCE",
    )
    for bar, val in zip(bars_left, [data["ece"], data["mce"]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(data["ece"], data["mce"]) * 0.05,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("ECE / MCE")
    ax.set_xticks([0, 1, 3])
    ax.set_xticklabels(["ECE", "MCE", "Brier"])
    ax.set_title("Calibration Metrics")
    y_max_left = max(data["ece"], data["mce"], 0.001) * 1.4
    ax.set_ylim(0, y_max_left)
    ax.grid(True, alpha=0.3, axis="y")

    # Eixo direito para Brier (escala diferente)
    ax2 = ax.twinx()
    bar_brier = ax2.bar(
        [3], [data["brier"]], width=0.6,
        color="#55a868", edgecolor="black", linewidth=0.5,
    )
    ax2.text(
        3, data["brier"] + data["brier"] * 0.05,
        f"{data['brier']:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
    ax2.set_ylabel("Brier Score")
    ax2.set_ylim(0, max(data["brier"], 0.01) * 1.3)

    # --- (1,1) Summary ---
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = (
        f"{'Perplexity:':<14s} {data['perplexity']:>10.2f}\n"
        f"{'Accuracy:':<14s} {data['accuracy']:>10.4f}\n"
        f"{'Confidence:':<14s} {data['mean_confidence']:>10.6f}\n"
        f"\n"
        f"{'ECE:':<14s} {data['ece']:>10.4f}\n"
        f"{'MCE:':<14s} {data['mce']:>10.4f}\n"
        f"{'Brier:':<14s} {data['brier']:>10.4f}"
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
