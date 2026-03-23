"""Avaliacao padronizada do DRM Transformer.

Metricas:
  - Perplexity em conjunto fixo (val split do baseline)
  - HellaSwag (TODO: requer modelo treinado em escala)
  - ARC-Easy (TODO: requer modelo treinado em escala)

Uso:
    # Perplexity no baseline val set
    python scripts/eval_standard.py \
        --checkpoint checkpoints/ablations/full/best.pt \
        --eval-data data/baseline/val

    # Avaliar todas as ablacoes
    python scripts/eval_standard.py --all-ablations

    # Com config especifica
    python scripts/eval_standard.py \
        --checkpoint checkpoints/baseline_1m/best.pt \
        --config configs/baselines/small_1m.yaml \
        --eval-data data/baseline/val
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml
from drm_transformer.config import DRMTransformerConfig
from drm_transformer.model import DRMTransformer
from drm_transformer.training.data import create_dataloader


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str = "",
    device: str = "cuda",
) -> tuple:
    """Carrega modelo de checkpoint.

    Args:
        checkpoint_path: Caminho do .pt
        config_path: Caminho do YAML (opcional, usa config do checkpoint)
        device: Device alvo

    Returns:
        (model, config_dict)
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = state.get("config", {})

    model_config = DRMTransformerConfig(
        vocab_size=config.get("vocab_size", 50000),
        max_seq_len=config.get("max_seq_len", 256),
        d_model=config.get("d_model", 64),
        n_layers=config.get("n_layers", 4),
        n_heads=config.get("n_heads", 2),
        d_ff=config.get("d_ff", 256),
        dropout=0.0,  # eval mode
        bias=config.get("bias", False),
        d_manifold=config.get("d_manifold", 4),
        metric_hidden=config.get("metric_hidden", 16),
        metric_rank=config.get("metric_rank", 4),
        n_quad=config.get("n_quad", 0),
        gamma_enabled=config.get("gamma_enabled", True),
        gamma_c=config.get("gamma_c", 2.0),
        gamma_alpha=config.get("gamma_alpha", 0.0),
        gravity_enabled=config.get("gravity_enabled", True),
        gravity_strength=config.get("gravity_strength", 0.1),
        gravity_n_rff=config.get("gravity_n_rff", 64),
        n_anchors=config.get("n_anchors", 6),
        temperature_init=config.get("temperature_init", 1.0),
        temperature_min=config.get("temperature_min", 0.5),
        variable_dim=config.get("variable_dim", True),
    )

    model = DRMTransformer(model_config)

    model_state = state.get("model", state)
    cleaned = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in model_state.items()
    }
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("[MODEL] %s params, device=%s", f"{total_params:,}", device)

    return model, config


@torch.no_grad()
def eval_perplexity(
    model: torch.nn.Module,
    eval_data: str,
    config: dict,
    device: str = "cuda",
    max_batches: int = 0,
) -> dict:
    """Calcula perplexity no eval set.

    Args:
        model: Modelo em eval mode.
        eval_data: Diretorio com shards .npy.
        config: Config dict.
        device: Device.
        max_batches: Limite de batches (0 = todos).

    Returns:
        Dict com loss, perplexity, n_batches, n_tokens.
    """
    loader = create_dataloader(
        data_dir=eval_data,
        seq_len=config.get("max_seq_len", 256),
        batch_size=config.get("batch_size", 16),
        rank=0,
        world_size=1,
    )

    total_loss = 0.0
    n_batches = 0
    n_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        logits, loss = model(input_ids, targets)
        total_loss += loss.item()
        n_tokens += targets.numel()
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))

    return {
        "eval_loss": round(avg_loss, 6),
        "perplexity": round(ppl, 2),
        "n_batches": n_batches,
        "n_tokens": n_tokens,
    }


def eval_single(checkpoint: str, eval_data: str, config_path: str, device: str) -> dict:
    """Avalia um unico checkpoint."""
    model, config = load_model_from_checkpoint(checkpoint, config_path, device)
    results = eval_perplexity(model, eval_data, config, device)
    results["checkpoint"] = checkpoint
    logger.info("[EVAL] loss=%.4f | ppl=%.2f | tokens=%d",
                results["eval_loss"], results["perplexity"], results["n_tokens"])
    return results


def eval_all_ablations(device: str) -> list:
    """Avalia todas as ablacoes que tem checkpoint."""
    ablation_names = ["full", "no_gravity", "no_gamma", "no_variable_dim"]
    results = []

    for name in ablation_names:
        ckpt = Path(f"checkpoints/ablations/{name}/best.pt")
        config_path = Path(f"configs/ablations/{name}.yaml")

        if not ckpt.exists():
            logger.warning("[SKIP] %s: checkpoint nao encontrado", name)
            continue

        logger.info("\n[ABLATION] %s", name)
        model, config = load_model_from_checkpoint(str(ckpt), str(config_path), device)
        r = eval_perplexity(model, "data/baseline/val", config, device)
        r["name"] = name
        r["checkpoint"] = str(ckpt)
        results.append(r)

        logger.info("  loss=%.4f | ppl=%.2f", r["eval_loss"], r["perplexity"])

        # Liberar memoria
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Avaliacao padronizada DRM Transformer")
    parser.add_argument("--checkpoint", default="", help="Caminho do .pt")
    parser.add_argument("--config", default="", help="Caminho do YAML")
    parser.add_argument("--eval-data", default="data/baseline/val", help="Diretorio de eval")
    parser.add_argument("--all-ablations", action="store_true", help="Avaliar todas as ablacoes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="", help="Salvar resultados em JSON")
    args = parser.parse_args()

    if args.all_ablations:
        results = eval_all_ablations(args.device)
    elif args.checkpoint:
        results = [eval_single(args.checkpoint, args.eval_data, args.config, args.device)]
    else:
        parser.error("Especifique --checkpoint ou --all-ablations")
        return

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("[SAVED] %s", args.output)
    elif results:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
