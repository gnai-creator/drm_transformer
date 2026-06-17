"""Diagnostico de ablacoes geometricas em runtime para um checkpoint DRM.

Compara o modelo completo contra variantes do mesmo checkpoint:
- U real vs U zero
- gravity on/off
- gamma on/off
- DimensionalGate on/off quando presente

Salva JSON com loss/PPL, diferencas de logits/attention e diagnosticos de
distancia. E uma avaliacao mecanistica, nao substitui ablations treinadas.
"""

import argparse
import dataclasses
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml

from drm_transformer.config import DRMTransformerConfig
from drm_transformer.model import DRMTransformer
from drm_transformer.training.data import create_dataloader


def load_model(checkpoint: str, config_path: str, device: str) -> DRMTransformer:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = yaml.safe_load(Path(config_path).read_text()) if config_path else state.get("config", {})
    valid = {f.name for f in dataclasses.fields(DRMTransformerConfig)}
    model = DRMTransformer(DRMTransformerConfig(**{k: v for k, v in (cfg or {}).items() if k in valid}))
    model_state = state.get("model", state)
    cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


@contextmanager
def no_change(_: DRMTransformer) -> Iterator[None]:
    yield


@contextmanager
def zero_metric_u(model: DRMTransformer) -> Iterator[None]:
    original = model.metric_net.forward

    def forward_zero(coords: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*coords.shape[:-1], model.metric_net.dim, model.metric_net.rank, device=coords.device, dtype=coords.dtype)

    model.metric_net.forward = forward_zero
    try:
        yield
    finally:
        model.metric_net.forward = original


@contextmanager
def gravity_off(model: DRMTransformer) -> Iterator[None]:
    original = model.gravity_field
    model.gravity_field = None
    try:
        yield
    finally:
        model.gravity_field = original


@contextmanager
def gamma_off(model: DRMTransformer) -> Iterator[None]:
    originals = [block.attn.gamma_enabled for block in model.blocks]
    for block in model.blocks:
        block.attn.gamma_enabled = False
    try:
        yield
    finally:
        for block, value in zip(model.blocks, originals):
            block.attn.gamma_enabled = value


@contextmanager
def dimensional_gate_off(model: DRMTransformer) -> Iterator[None]:
    original = model.dim_gate
    model.dim_gate = None
    try:
        yield
    finally:
        model.dim_gate = original


def attention_entropy(attn: torch.Tensor) -> float:
    p = attn.clamp_min(1e-8)
    return float((-(p * p.log()).sum(dim=-1)).mean().item())


def distance_correlation(components: Dict[str, torch.Tensor]) -> float:
    if "dist_sq" not in components or "dist_euc" not in components:
        return float("nan")
    x = components["dist_sq"].flatten().float()
    y = components["dist_euc"].flatten().float()
    if x.numel() < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return float("nan")
    corr = torch.corrcoef(torch.stack([x, y]))[0, 1]
    return float(corr.item())


@torch.no_grad()
def eval_variant(
    model: DRMTransformer,
    loader,
    device: str,
    mutate: Callable[[DRMTransformer], Iterator[None]],
    baseline_cache: list | None = None,
    max_batches: int = 10,
) -> tuple[dict, list]:
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    comparisons = []
    cache = []
    last_diagnostics = {}
    last_components = {}
    last_attn = None

    with mutate(model):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            logits, loss = model(input_ids, targets)
            attn = model.blocks[0].attn.last_attention
            diagnostics = model.blocks[0].attn.last_distance_diagnostics
            components = model.blocks[0].attn.last_distance_components

            total_loss += float(loss.item())
            total_tokens += int(targets.numel())
            n_batches += 1

            entry = {
                "logits": logits.detach().cpu(),
                "attention": attn.detach().cpu() if attn is not None else None,
            }
            cache.append(entry)

            if baseline_cache is not None and batch_idx < len(baseline_cache):
                base = baseline_cache[batch_idx]
                logit_l2 = torch.linalg.norm(logits.detach().cpu() - base["logits"]).item() / max(logits.numel(), 1)
                attn_l2 = None
                attn_kl = None
                if attn is not None and base["attention"] is not None:
                    p = base["attention"].clamp_min(1e-8)
                    q = attn.detach().cpu().clamp_min(1e-8)
                    attn_l2 = torch.linalg.norm(q - p).item() / max(q.numel(), 1)
                    attn_kl = (p * (p.log() - q.log())).sum(dim=-1).mean().item()
                comparisons.append({"logit_l2": logit_l2, "attention_l2": attn_l2, "attention_kl": attn_kl})

            if n_batches >= max_batches:
                last_diagnostics = diagnostics
                last_components = components
                last_attn = attn
                break

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {
        "eval_loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20)),
        "n_batches": n_batches,
        "n_tokens": total_tokens,
        "attention_entropy": attention_entropy(last_attn.detach().cpu()) if last_attn is not None else None,
        "geodesic_euclidean_correlation": distance_correlation(last_components),
    }
    for key, value in last_diagnostics.items():
        if torch.is_tensor(value):
            metrics[key] = float(value.cpu().item())

    if comparisons:
        for key in ("logit_l2", "attention_l2", "attention_kl"):
            vals = [c[key] for c in comparisons if c[key] is not None]
            metrics[key] = sum(vals) / max(len(vals), 1) if vals else None
    return metrics, cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime geometry ablation diagnostics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config, args.device)
    loader = create_dataloader(
        data_dir=args.eval_data,
        seq_len=model.config.max_seq_len,
        batch_size=1,
        rank=0,
        world_size=1,
    )

    results = {}
    results["full"], baseline_cache = eval_variant(model, loader, args.device, no_change, None, args.max_batches)

    variants = {
        "u_zero": zero_metric_u,
        "no_gravity_runtime": gravity_off,
        "no_gamma_runtime": gamma_off,
        "no_variable_dim_runtime": dimensional_gate_off,
    }
    for name, mutator in variants.items():
        loader = create_dataloader(
            data_dir=args.eval_data,
            seq_len=model.config.max_seq_len,
            batch_size=1,
            rank=0,
            world_size=1,
        )
        results[name], _ = eval_variant(model, loader, args.device, mutator, baseline_cache, args.max_batches)

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "eval_data": args.eval_data,
        "results": results,
        "note": "Runtime ablations measure immediate module influence; retrained ablations are still required for capability claims.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
