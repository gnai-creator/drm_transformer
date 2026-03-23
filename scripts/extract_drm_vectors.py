"""Extrai vectores DRM (coords, G_diag, gamma, mass) do modelo treinado.

Faz forward pass capturando coordenadas do manifold via hook no primeiro
layer de attention (q_to_manifold + sigmoid), e computa G(x), gamma e mass.

Uso:
    python scripts/extract_drm_vectors.py \
        --checkpoint checkpoints/1m/final.pt \
        --data-dir data/ \
        --output-dir eval_results/foliation_1m \
        --max-tokens 100000
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from drm_transformer import DRMTransformerConfig, DRMTransformer
from drm_transformer.manifold import gamma_scale
from drm_transformer.training.data import ShardedDataset

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str) -> DRMTransformer:
    """Carrega modelo a partir de checkpoint.

    Args:
        checkpoint_path: Caminho para o checkpoint .pt.
        device: Device para carregar.

    Returns:
        Modelo carregado em eval mode.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = state.get("config", {})
    config = DRMTransformerConfig(**{
        k: v for k, v in config_dict.items()
        if hasattr(DRMTransformerConfig, k)
    })

    model = DRMTransformer(config)

    model_state = state.get("model", state)
    cleaned = {}
    for k, v in model_state.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()

    logger.info(
        "[LOAD] %s (d_manifold=%d, step=%s)",
        checkpoint_path,
        config.d_manifold,
        state.get("global_step", "?"),
    )
    return model


def extract_vectors(
    model: DRMTransformer,
    dataset: ShardedDataset,
    max_seqs: int = 2000,
    batch_size: int = 4,
    device: str = "cuda",
) -> dict:
    """Extrai coordenadas do manifold, G_diag, gamma e mass.

    Usa hook no primeiro layer de attention para capturar q_manifold.

    Args:
        model: Modelo DRM treinado.
        dataset: Dataset tokenizado.
        max_seqs: Numero maximo de sequencias.
        batch_size: Tamanho do batch.
        device: Device.

    Returns:
        Dict com arrays numpy: coords, G_diag, gamma, mass, token_ids.
    """
    config = model.config
    d_manifold = config.d_manifold

    all_coords = []
    all_G_diag = []
    all_gamma = []
    all_mass = []
    all_token_ids = []

    captured_coords = {}

    def hook_fn(module, input_args, output):
        """Captura q_manifold do primeiro attention layer."""
        # Nao podemos hookar q_to_manifold diretamente porque e um Linear.
        # Em vez disso, hookamos o forward completo do DRMAttention e
        # recomputamos q_manifold. Isso e seguro em eval mode.
        pass

    n_seqs = min(max_seqs, len(dataset))
    n_batches = (n_seqs + batch_size - 1) // batch_size

    logger.info("[EXTRACT] %d seqs, batch=%d, device=%s", n_seqs, batch_size, device)
    t0 = time.time()

    with torch.no_grad():
        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, n_seqs)
            actual_bs = end - start

            batch_ids = []
            for i in range(start, end):
                sample = dataset[i]
                batch_ids.append(sample["input_ids"])

            input_ids = torch.stack(batch_ids).to(device)
            B, T = input_ids.shape

            # Forward ate obter embeddings
            x = model.token_emb(input_ids)
            x = model.emb_dropout(x)

            if model.dim_gate is not None:
                x, _ = model.dim_gate(x)

            # Extrair coords do primeiro block: usar q_to_manifold
            block0 = model.blocks[0]
            attn = block0.attn

            # Pre-norm (como no block forward)
            x_normed = block0.norm1(x)

            # Q projection
            q = attn.q_proj(x_normed)
            q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)

            # Projetar para manifold (head 0 para simplicidade)
            q_manifold = torch.sigmoid(attn.q_to_manifold(q[:, 0]))  # [B, T, d_manifold]

            coords = q_manifold  # [B, T, d_manifold]

            # G(x) via MetricNet
            coords_flat = coords.reshape(-1, d_manifold)
            G = model.metric_net(coords_flat)  # [B*T, D, D]
            G_diag = G.diagonal(dim1=-2, dim2=-1)  # [B*T, D]
            G_diag = G_diag.reshape(B, T, d_manifold)

            # Gamma
            gamma = gamma_scale(
                coords, model.anchors, c_param=config.gamma_c,
            )  # [B, T, 1]

            # Mass (se gravity habilitado)
            if model.gravity_field is not None:
                mass = model.gravity_field.compute_mass(coords)  # [B, T, 1]
            else:
                mass = torch.zeros(B, T, 1, device=device)

            # Coletar
            all_coords.append(coords.reshape(-1, d_manifold).cpu().numpy())
            all_G_diag.append(G_diag.reshape(-1, d_manifold).cpu().numpy())
            all_gamma.append(gamma.reshape(-1, 1).cpu().numpy())
            all_mass.append(mass.reshape(-1, 1).cpu().numpy())
            all_token_ids.append(input_ids.reshape(-1).cpu().numpy())

            if (b_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (b_idx + 1) * batch_size * T / elapsed
                logger.info(
                    "  [%d/%d] %.0f tok/s, %d tokens",
                    b_idx + 1, n_batches, rate,
                    (b_idx + 1) * actual_bs * T,
                )

    elapsed = time.time() - t0
    coords_all = np.concatenate(all_coords, axis=0)
    logger.info(
        "[EXTRACT] %d tokens em %.1fs (%.0f tok/s)",
        len(coords_all), elapsed, len(coords_all) / elapsed,
    )

    return {
        "coords": coords_all,
        "G_diag": np.concatenate(all_G_diag, axis=0),
        "gamma": np.concatenate(all_gamma, axis=0),
        "mass": np.concatenate(all_mass, axis=0),
        "token_ids": np.concatenate(all_token_ids, axis=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Extrai vectores DRM do modelo")
    parser.add_argument("--checkpoint", required=True, help="Caminho para checkpoint .pt")
    parser.add_argument("--data-dir", required=True, help="Diretorio com shards tokenizados")
    parser.add_argument("--output-dir", default="eval_results/foliation", help="Diretorio de saida")
    parser.add_argument("--max-tokens", type=int, default=1_000_000, help="Maximo de tokens")
    parser.add_argument("--max-seqs", type=int, default=2000, help="Maximo de sequencias")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label", default="drm", help="Prefixo dos arquivos de saida")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    model = load_model(args.checkpoint, args.device)
    config = model.config

    dataset = ShardedDataset(
        args.data_dir,
        seq_len=config.max_seq_len,
        max_tokens=args.max_tokens,
    )
    logger.info("[DATA] %d seqs de %d tokens", len(dataset), config.max_seq_len)

    vectors = extract_vectors(
        model, dataset,
        max_seqs=args.max_seqs,
        batch_size=args.batch_size,
        device=args.device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = args.label
    np.save(out_dir / f"{label}_coords.npy", vectors["coords"])
    np.save(out_dir / f"{label}_G_diag.npy", vectors["G_diag"])
    np.save(out_dir / f"{label}_gamma.npy", vectors["gamma"])
    np.save(out_dir / f"{label}_mass.npy", vectors["mass"])
    np.save(out_dir / f"{label}_token_ids.npy", vectors["token_ids"])

    metadata = {
        "checkpoint": args.checkpoint,
        "label": label,
        "n_vectors": len(vectors["coords"]),
        "d_manifold": config.d_manifold,
        "gamma_enabled": config.gamma_enabled,
        "gamma_c": config.gamma_c,
        "gravity_enabled": config.gravity_enabled,
        "max_seq_len": config.max_seq_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / f"{label}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[SAVE] %s/ (%d vectors, d=%d)", out_dir, len(vectors["coords"]), config.d_manifold)


if __name__ == "__main__":
    main()
