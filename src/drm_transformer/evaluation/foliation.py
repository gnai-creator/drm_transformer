"""Avaliador de foliation topologica para DRM Transformer.

Encapsula o pipeline completo: extracao de vectores + Voronoi foliation.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from drm_transformer import DRMTransformer
from drm_transformer.manifold import gamma_scale

logger = logging.getLogger(__name__)


class DRMFoliationEvaluator:
    """Pipeline completo de avaliacao topologica.

    Extrai coordenadas do manifold, computa G(x), gamma e mass,
    e aplica Voronoi foliation para detectar estrutura topologica.

    Args:
        model: Modelo DRM treinado.
        device: Device para computacao.
    """

    def __init__(self, model: DRMTransformer, device: str = "cuda"):
        self.model = model
        self.config = model.config
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract_vectors(
        self,
        dataloader: DataLoader,
        max_tokens: int = 500_000,
    ) -> dict:
        """Extrai vectores DRM do modelo.

        Args:
            dataloader: DataLoader com batches de input_ids.
            max_tokens: Maximo de tokens a processar.

        Returns:
            Dict com coords, G_diag, gamma, mass (numpy arrays).
        """
        d = self.config.d_manifold
        all_coords, all_G_diag, all_gamma, all_mass = [], [], [], []
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            B, T = input_ids.shape

            x = self.model.token_emb(input_ids)
            x = self.model.emb_dropout(x)

            if self.model.dim_gate is not None:
                x, _ = self.model.dim_gate(x)

            block0 = self.model.blocks[0]
            x_normed = block0.norm1(x)
            q = block0.attn.q_proj(x_normed)
            q = q.view(B, T, block0.attn.n_heads, block0.attn.d_head).transpose(1, 2)
            coords = torch.sigmoid(block0.attn.q_to_manifold(q[:, 0]))

            coords_flat = coords.reshape(-1, d)
            G = self.model.metric_net(coords_flat)
            G_diag = G.diagonal(dim1=-2, dim2=-1).reshape(B, T, d)

            gamma = gamma_scale(coords, self.model.anchors, c_param=self.config.gamma_c)

            mass = torch.zeros(B, T, 1, device=self.device)
            if self.model.gravity_field is not None:
                mass = self.model.gravity_field.compute_mass(coords)

            all_coords.append(coords.reshape(-1, d).cpu().numpy())
            all_G_diag.append(G_diag.reshape(-1, d).cpu().numpy())
            all_gamma.append(gamma.reshape(-1, 1).cpu().numpy())
            all_mass.append(mass.reshape(-1, 1).cpu().numpy())

            total_tokens += B * T
            if total_tokens >= max_tokens:
                break

        return {
            "coords": np.concatenate(all_coords),
            "G_diag": np.concatenate(all_G_diag),
            "gamma": np.concatenate(all_gamma),
            "mass": np.concatenate(all_mass),
        }

    def compute_foliation(
        self,
        vectors: dict,
        n_seeds: int = 30,
        homology_points: int = 1500,
    ) -> dict:
        """Computa pipeline de foliation sobre vectores extraidos.

        Args:
            vectors: Dict com coords, G_diag, gamma.
            n_seeds: Numero de clusters.
            homology_points: Subsample para homologia.

        Returns:
            Dict com foliation_score, topology, ARI, coherence.
        """
        # Import local para evitar dependencia obrigatoria
        import sys
        import os
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
        sys.path.insert(0, os.path.abspath(scripts_dir))

        from voronoi_foliation_drm import (
            voronoi_tessellation, ltsa_per_cell, tangent_coherence,
            compute_homology, stability_test, foliation_score,
        )

        coords = vectors["coords"]
        D = coords.shape[1]

        vor = voronoi_tessellation(coords, n_seeds=n_seeds)
        ltsa = ltsa_per_cell(coords, vor["labels"])
        coh = tangent_coherence(
            vor["centers"], ltsa["eigenvectors"], ltsa["eff_dims"],
        )
        hom = compute_homology(
            coords, G_diag=vectors.get("G_diag"),
            max_points=homology_points,
        )
        stab = stability_test(coords, n_seeds=n_seeds)
        F = foliation_score(
            ltsa["eff_dims"], coh["coherent_fraction"],
            stab["mean_ari"], D,
        )

        return {
            "foliation_score": F,
            "topology": hom.get("topology", "unknown"),
            "t2_valid": hom.get("t2_valid", False),
            "mean_ari": stab["mean_ari"],
            "coherent_fraction": coh["coherent_fraction"],
            "mean_eff_dim": float(ltsa["eff_dims"].mean()),
            "homology": hom.get("homology", {}),
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        max_tokens: int = 500_000,
        n_seeds: int = 30,
        homology_points: int = 1500,
    ) -> dict:
        """Pipeline completo: extract + foliation.

        Args:
            dataloader: DataLoader com input_ids.
            max_tokens: Maximo de tokens.
            n_seeds: Clusters.
            homology_points: Subsample homologia.

        Returns:
            Dict completo de resultados.
        """
        t0 = time.time()

        vectors = self.extract_vectors(dataloader, max_tokens=max_tokens)
        results = self.compute_foliation(
            vectors, n_seeds=n_seeds,
            homology_points=homology_points,
        )

        results["n_vectors"] = len(vectors["coords"])
        results["d_manifold"] = self.config.d_manifold
        results["elapsed_s"] = time.time() - t0

        logger.info(
            "[FOLIATION] F=%.4f, topology=%s, ARI=%.4f (%.1fs)",
            results["foliation_score"],
            results["topology"],
            results["mean_ari"],
            results["elapsed_s"],
        )

        return results
