"""Probe simples para medir se anchors geometricos separam labels externas.

Entrada JSONL:
    {"text": "...", "label": "truth"}

O script usa uma tokenizacao byte-level deterministica como fallback, porque o
repositorio nao fixa um tokenizer textual. A avaliacao e diagnostica: nearest
anchor nao valida semantica por si so.
"""

import argparse
import dataclasses
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml

from drm_transformer.config import DRMTransformerConfig
from drm_transformer.model import DRMTransformer


ANCHOR_NAMES = ["truth", "ignorance", "safety", "complexity", "creativity", "grounding"]


def _load_config(path: str, checkpoint_state: dict) -> dict:
    if path:
        return yaml.safe_load(Path(path).read_text()) or {}
    return checkpoint_state.get("config", {})


def _model_config(config: dict) -> DRMTransformerConfig:
    valid = {f.name for f in dataclasses.fields(DRMTransformerConfig)}
    model_cfg = {k: v for k, v in config.items() if k in valid}
    return DRMTransformerConfig(**model_cfg)


def load_model(checkpoint: str, config_path: str, device: str) -> DRMTransformer:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _load_config(config_path, state)
    model = DRMTransformer(_model_config(cfg))
    model_state = state.get("model", state)
    cleaned = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def encode_text(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    ids = [b % vocab_size for b in text.encode("utf-8", errors="ignore")]
    ids = ids[:seq_len]
    if not ids:
        ids = [0]
    ids = ids + [0] * (seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def mean_coords(model: DRMTransformer, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.token_emb(input_ids)
    if model.dim_gate is not None:
        x, _ = model.dim_gate(x)
    block0 = model.blocks[0]
    x = block0.norm1(x)
    B, T = input_ids.shape
    q = block0.attn.q_proj(x).view(B, T, block0.attn.n_heads, block0.attn.d_head).transpose(1, 2)
    coords = torch.sigmoid(block0.attn.q_to_manifold(q[:, 0]))
    return coords.mean(dim=1).squeeze(0)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia separacao nearest-anchor em JSONL rotulado")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.config, args.device)
    labels = ANCHOR_NAMES[: model.anchors.shape[0]]
    rows = read_jsonl(args.jsonl)

    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_label_distances: Dict[str, List[List[float]]] = defaultdict(list)
    correct = 0
    examples = []

    for row in rows:
        label = row["label"]
        ids = encode_text(row["text"], model.config.vocab_size, min(args.seq_len, model.config.max_seq_len)).to(args.device)
        coords = mean_coords(model, ids)
        distances = torch.linalg.norm(model.anchors - coords.unsqueeze(0), dim=-1)
        pred_idx = int(distances.argmin().item())
        pred = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        correct += int(pred == label)
        confusion[label][pred] += 1
        per_label_distances[label].append(distances.cpu().tolist())
        examples.append({"label": label, "prediction": pred, "distances": distances.cpu().tolist()})

    separation = {}
    for label, values in per_label_distances.items():
        d = torch.tensor(values)
        separation[label] = {
            "mean_distance_to_anchors": d.mean(dim=0).tolist(),
            "margin_nearest_minus_label": None,
        }
        if label in labels:
            label_idx = labels.index(label)
            nearest_other = torch.cat([d[:, :label_idx], d[:, label_idx + 1 :]], dim=1).min(dim=1).values
            separation[label]["margin_nearest_minus_label"] = (nearest_other - d[:, label_idx]).mean().item()

    result = {
        "checkpoint": args.checkpoint,
        "jsonl": args.jsonl,
        "n_examples": len(rows),
        "anchor_names": labels,
        "accuracy_nearest_anchor": correct / max(len(rows), 1),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "separation_by_label": separation,
        "examples": examples,
        "note": "Anchors are geometric priors; this probe is evidence only for the supplied labeled set.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
