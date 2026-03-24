"""Pipeline de Voronoi Foliation para DRM Transformer.

9 fases: Voronoi, LTSA, Coherence, Reeb, Homology, Null Models,
Stability, Foliation Score, Coords Correlation.

Uso:
    python scripts/voronoi_foliation_drm.py \
        --coords eval_results/foliation/drm_coords.npy \
        --output-dir eval_results/foliation \
        --n-seeds 30 \
        --homology-points 1500
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# ============================================================
# FASE 1: Voronoi Tessellation
# ============================================================

def voronoi_tessellation(
    coords: np.ndarray,
    n_seeds: int = 30,
    G_diag: np.ndarray = None,
    gamma: np.ndarray = None,
    use_gamma_distance: bool = False,
) -> dict:
    """KMeans clustering sobre coords do manifold.

    Args:
        coords: [N, D] coordenadas do manifold.
        n_seeds: Numero de clusters.
        G_diag: [N, D] diagonal de G(x) (opcional, para distancia ponderada).
        gamma: [N, 1] fator gamma (opcional).
        use_gamma_distance: Ponderar distancia por gamma.

    Returns:
        Dict com labels, centers, cell_sizes.
    """
    logger.info("[FASE 1] Voronoi Tessellation (K=%d, N=%d)", n_seeds, len(coords))

    km = KMeans(n_clusters=n_seeds, n_init=10, random_state=42, max_iter=100)
    labels = km.fit_predict(coords)
    centers = km.cluster_centers_

    sizes = np.bincount(labels, minlength=n_seeds)
    logger.info(
        "  Cells: min=%d, median=%d, max=%d",
        sizes.min(), int(np.median(sizes)), sizes.max(),
    )

    return {"labels": labels, "centers": centers, "cell_sizes": sizes}


# ============================================================
# FASE 2: Local Tangent Space Analysis (LTSA)
# ============================================================

def ltsa_per_cell(
    coords: np.ndarray,
    labels: np.ndarray,
    min_points: int = 100,
    threshold: float = 0.95,
) -> dict:
    """PCA por celula Voronoi para dimensao efectiva.

    Args:
        coords: [N, D] coordenadas.
        labels: [N] cluster labels.
        min_points: Minimo de pontos por celula.
        threshold: Variancia cumulativa para eff_dim.

    Returns:
        Dict com eigenvalues, eff_dims, cell_sizes.
    """
    logger.info("[FASE 2] Local Tangent Space Analysis")

    D = coords.shape[1]
    n_cells = labels.max() + 1

    eigenvalues = np.zeros((n_cells, D))
    eigenvectors = np.zeros((n_cells, D, D))
    eff_dims = np.zeros(n_cells, dtype=int)
    cell_sizes = np.bincount(labels, minlength=n_cells)

    for k in range(n_cells):
        mask = labels == k
        if mask.sum() < min_points:
            eff_dims[k] = D
            eigenvalues[k] = np.ones(D) / D
            eigenvectors[k] = np.eye(D)
            continue

        X = coords[mask]
        X_c = X - X.mean(axis=0)
        cov = X_c.T @ X_c / (len(X_c) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigenvalues[k] = eigvals
        eigenvectors[k] = eigvecs.T

        total = eigvals.sum()
        if total < 1e-12:
            eff_dims[k] = 1
            continue
        cumvar = np.cumsum(eigvals) / total
        eff_dims[k] = int(np.searchsorted(cumvar, threshold) + 1)

    logger.info(
        "  eff_dims: mean=%.1f, median=%.1f, dist=%s",
        eff_dims.mean(), np.median(eff_dims),
        np.bincount(eff_dims, minlength=D + 1)[1:].tolist(),
    )

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "eff_dims": eff_dims,
        "cell_sizes": cell_sizes,
    }


# ============================================================
# FASE 3: Tangent Coherence
# ============================================================

def tangent_coherence(
    centers: np.ndarray,
    eigenvectors: np.ndarray,
    eff_dims: np.ndarray,
    n_neighbors: int = 5,
) -> dict:
    """Angulo entre espacos tangentes de celulas adjacentes.

    Args:
        centers: [K, D] centroides.
        eigenvectors: [K, D, D] eigenvectors por celula.
        eff_dims: [K] dimensao efectiva.
        n_neighbors: Vizinhos para adjacencia.

    Returns:
        Dict com coherence_matrix, mean_angle, coherent_fraction.
    """
    logger.info("[FASE 3] Tangent Coherence")

    K, D = centers.shape[0], centers.shape[1]
    nn_model = NearestNeighbors(n_neighbors=min(n_neighbors + 1, K))
    nn_model.fit(centers)
    _, indices = nn_model.kneighbors(centers)

    coherence_matrix = np.full((K, K), np.nan)
    angles = []

    for i in range(K):
        di = max(eff_dims[i], 1)
        Si = eigenvectors[i, :di, :]  # [di, D]

        for j in indices[i, 1:]:
            dj = max(eff_dims[j], 1)
            Sj = eigenvectors[j, :dj, :]

            # Angulo principal via SVD
            M = Si @ Sj.T
            svd_vals = np.linalg.svd(M, compute_uv=False)
            cos_angle = np.clip(svd_vals[0], -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            coherence_matrix[i, j] = angle
            angles.append(angle)

    angles = np.array(angles) if angles else np.array([90.0])
    coherent = (angles < 30.0).mean()

    logger.info(
        "  mean_angle=%.1f deg, coherent_fraction=%.3f",
        angles.mean(), coherent,
    )

    return {
        "coherence_matrix": coherence_matrix,
        "mean_angle": float(angles.mean()),
        "median_angle": float(np.median(angles)),
        "coherent_fraction": float(coherent),
    }


# ============================================================
# FASE 4: Reeb Graph
# ============================================================

def compute_reeb_graph(
    coords: np.ndarray,
    labels: np.ndarray,
    func_values: np.ndarray,
    func_name: str = "gamma",
    n_levels: int = 50,
    n_neighbors: int = 15,
    std_threshold: float = 0.03  # era 0.15
) -> dict:
    """Grafo de Reeb via level sets.

    Args:
        coords: [N, D] coordenadas.
        labels: [N] cluster labels.
        func_values: [N] funcao escalar para level sets.
        func_name: Nome da funcao.
        n_levels: Numero de niveis.
        n_neighbors: Vizinhos para conectividade.
        std_threshold: Std minimo para computar.

    Returns:
        Dict com n_nodes, n_edges, n_splits, n_merges.
    """
    logger.info("[FASE 4] Reeb Graph (%s)", func_name)

    f = func_values.copy()
    f_std = f.std()

    if f_std < std_threshold:
        logger.info("  [SKIP] std=%.4f < %.2f, funcao degenerada", f_std, std_threshold)
        return {
            "func_name": func_name,
            "skipped": True,
            "f_std": float(f_std),
        }

    # Logit transform se saturado
    transform = "none"
    if f_std < 0.2:
        f_clip = np.clip(f, 0.001, 0.999)
        f = np.log(f_clip / (1.0 - f_clip))
        transform = "logit"

    f_min, f_max = f.min(), f.max()
    edges_list = np.linspace(f_min, f_max, n_levels + 1)

    # Subsample para NearestNeighbors
    N = len(coords)
    sub_n = min(N, 30000)
    if N > sub_n:
        sub_idx = np.random.default_rng(42).choice(N, sub_n, replace=False)
        coords_sub = coords[sub_idx]
        f_sub = f[sub_idx]
        labels_sub = labels[sub_idx]
    else:
        coords_sub = coords
        f_sub = f
        labels_sub = labels
        sub_idx = np.arange(N)

    nn_model = NearestNeighbors(n_neighbors=min(n_neighbors, len(coords_sub)))
    nn_model.fit(coords_sub)

    from scipy.sparse.csgraph import connected_components as cc
    from scipy.sparse import lil_matrix

    nodes = []
    prev_components = {}

    n_splits = 0
    n_merges = 0
    edge_list = []

    for lv in range(n_levels):
        lo, hi = edges_list[lv], edges_list[lv + 1]
        mask = (f_sub >= lo) & (f_sub < hi) if lv < n_levels - 1 else (f_sub >= lo) & (f_sub <= hi)
        idx_in_level = np.where(mask)[0]

        if len(idx_in_level) < 2:
            continue

        # kNN subgraph
        dists, nbrs = nn_model.kneighbors(coords_sub[idx_in_level])
        adj = lil_matrix((len(idx_in_level), len(idx_in_level)), dtype=bool)

        local_map = {g: i for i, g in enumerate(idx_in_level)}
        for i_local, (i_global, i_nbrs) in enumerate(zip(idx_in_level, nbrs)):
            for nb in i_nbrs:
                if nb in local_map:
                    adj[i_local, local_map[nb]] = True

        n_comp, comp_labels = cc(adj, directed=False)

        curr_components = {}
        for c in range(n_comp):
            c_mask = comp_labels == c
            n_pts = c_mask.sum()
            if n_pts < 1:
                continue
            node_id = len(nodes)
            nodes.append({
                "level": lv,
                "level_value": float((lo + hi) / 2),
                "n_points": int(n_pts),
            })
            curr_components[node_id] = set(idx_in_level[c_mask])

        # Conectar com nivel anterior
        for curr_id, curr_pts in curr_components.items():
            for prev_id, prev_pts in prev_components.items():
                # Amostra para checar adjacencia
                sample_curr = list(curr_pts)[:50]
                sample_prev = list(prev_pts)[:50]
                connected = False
                for sc in sample_curr:
                    _, sc_nbrs = nn_model.kneighbors(coords_sub[sc:sc+1])
                    if any(nb in prev_pts for nb in sc_nbrs[0]):
                        connected = True
                        break
                if connected:
                    edge_list.append((prev_id, curr_id))

        # Splits e merges
        in_degree = {}
        out_degree = {}
        for p, c in edge_list:
            out_degree[p] = out_degree.get(p, 0) + 1
            in_degree[c] = in_degree.get(c, 0) + 1

        prev_components = curr_components

    n_splits = sum(1 for d in out_degree.values() if d > 1) if 'out_degree' in dir() else 0
    n_merges = sum(1 for d in in_degree.values() if d > 1) if 'in_degree' in dir() else 0

    result = {
        "func_name": func_name,
        "skipped": False,
        "transform": transform,
        "n_nodes": len(nodes),
        "n_edges": len(edge_list),
        "n_splits": n_splits,
        "n_merges": n_merges,
        "is_tree": n_splits == 0 and n_merges == 0,
        "f_std_raw": float(func_values.std()),
        "f_std_final": float(f.std()),
    }

    logger.info(
        "  nodes=%d, edges=%d, splits=%d, merges=%d",
        result["n_nodes"], result["n_edges"],
        result["n_splits"], result["n_merges"],
    )

    return result


# ============================================================
# FASE 5: Persistent Homology
# ============================================================

def compute_homology(
    coords: np.ndarray,
    G_diag: np.ndarray = None,
    max_points: int = 1500,
    max_dim: int = 2,
) -> dict:
    """Persistent homology (H0, H1, H2) via ripser.

    Args:
        coords: [N, D] coordenadas.
        G_diag: [N, D] diagonal de G(x) para distancia ponderada.
        max_points: Subsample maximo.
        max_dim: Dimensao maxima da homologia.

    Returns:
        Dict com H0, H1, H2, topology label.
    """
    logger.info("[FASE 5] Persistent Homology (max_points=%d)", max_points)

    try:
        from ripser import ripser
    except ImportError:
        logger.warning("  [SKIP] ripser nao instalado")
        return {"skipped": True}

    N = len(coords)
    if N > max_points:
        idx = np.random.default_rng(42).choice(N, max_points, replace=False)
        pts = coords[idx]
        g_diag = G_diag[idx] if G_diag is not None else None
    else:
        pts = coords
        g_diag = G_diag

    # Distancia: se temos G_diag, usar Mahalanobis diagonal
    if g_diag is not None:
        n = len(pts)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            delta = pts[i] - pts[i+1:]
            # G_mean = media das diagonais dos dois pontos
            g_mean = (g_diag[i] + g_diag[i+1:]) / 2.0
            # d^2 = sum(delta^2 * g_mean) por eixo
            d_sq = (delta ** 2 * g_mean).sum(axis=-1)
            dist_matrix[i, i+1:] = np.sqrt(np.maximum(d_sq, 0))
            dist_matrix[i+1:, i] = dist_matrix[i, i+1:]

        result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    else:
        result = ripser(pts, maxdim=max_dim)

    diagrams = result["dgms"]

    homology = {}
    for d in range(min(len(diagrams), max_dim + 1)):
        dgm = diagrams[d]
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            homology[f"H{d}"] = {"n_features": 0, "long_bars": 0}
            continue

        persistence = finite[:, 1] - finite[:, 0]
        med = np.median(persistence)
        std = persistence.std()
        threshold = med + 2 * std
        long_bars = int((persistence > threshold).sum())

        homology[f"H{d}"] = {
            "n_features": len(persistence),
            "long_bars": long_bars,
            "max_persistence": float(persistence.max()),
            "mean_persistence": float(persistence.mean()),
        }

    h1_long = homology.get("H1", {}).get("long_bars", 0)
    h2_long = homology.get("H2", {}).get("long_bars", 0)

    if h1_long == 2 and h2_long == 1:
        topology = "torus T^2 (validated)"
        t2_valid = True
    elif h1_long == 0 and h2_long == 0:
        topology = "trivial"
        t2_valid = False
    elif h1_long == 1 and h2_long == 0:
        topology = "cylinder (S^1 x R)"
        t2_valid = False
    elif h1_long == 2 and h2_long == 0:
        topology = "partial T^2 (H2 mismatch)"
        t2_valid = False
    elif h1_long > 2:
        topology = f"genus > 1 (H1={h1_long})"
        t2_valid = False
    else:
        topology = f"other (H1={h1_long}, H2={h2_long})"
        t2_valid = False

    logger.info(
        "  H1=%d long, H2=%d long -> %s",
        h1_long, h2_long, topology,
    )

    return {
        "skipped": False,
        "homology": homology,
        "topology": topology,
        "t2_valid": t2_valid,
    }


# ============================================================
# FASE 6: Null Models
# ============================================================

def null_model(
    coords: np.ndarray,
    centers: np.ndarray,
    null_type: str = "shuffled",
) -> dict:
    """Null model para comparacao.

    Args:
        coords: [N, D] coordenadas originais.
        centers: [K, D] centroides originais.
        null_type: "shuffled" ou "uniform".

    Returns:
        Dict com metricas do null model.
    """
    logger.info("[FASE 6] Null model: %s", null_type)

    D = coords.shape[1]
    N = len(coords)

    if null_type == "shuffled":
        null_coords = coords.copy()
        rng = np.random.default_rng(42)
        for d in range(D):
            rng.shuffle(null_coords[:, d])
    elif null_type == "uniform":
        null_coords = np.random.default_rng(42).random((N, D))
    else:
        return {"type": null_type, "error": "unknown null_type"}

    # Recluster
    dists = cdist(null_coords, centers)
    null_labels = dists.argmin(axis=1)

    # LTSA
    ltsa = ltsa_per_cell(null_coords, null_labels, min_points=50)

    # Coherence
    coh = tangent_coherence(
        centers, ltsa["eigenvectors"], ltsa["eff_dims"],
    )

    # Entropy de eff_dims
    counts = np.bincount(ltsa["eff_dims"], minlength=D + 1)[1:]
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))

    return {
        "type": null_type,
        "mean_coherence_angle": coh["mean_angle"],
        "coherent_fraction": coh["coherent_fraction"],
        "dim_entropy": float(entropy),
        "mean_eff_dim": float(ltsa["eff_dims"].mean()),
    }


# ============================================================
# FASE 7: Stability (ARI)
# ============================================================

def stability_test(
    coords: np.ndarray,
    n_seeds: int = 30,
    n_restarts: int = 10,
) -> dict:
    """Adjusted Rand Index entre re-runs de KMeans.

    Args:
        coords: [N, D] coordenadas.
        n_seeds: Numero de clusters.
        n_restarts: Numero de re-runs.

    Returns:
        Dict com mean_ari, std_ari.
    """
    logger.info("[FASE 7] Stability (n_restarts=%d)", n_restarts)

    all_labels = []
    for r in range(n_restarts):
        km = KMeans(n_clusters=n_seeds, n_init=1, random_state=r * 42)
        all_labels.append(km.fit_predict(coords))

    aris = []
    for i in range(n_restarts):
        for j in range(i + 1, n_restarts):
            aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    aris = np.array(aris)
    logger.info("  mean_ari=%.4f, std=%.4f", aris.mean(), aris.std())

    return {
        "mean_ari": float(aris.mean()),
        "std_ari": float(aris.std()),
        "n_restarts": n_restarts,
    }


# ============================================================
# FASE 8: Foliation Score
# ============================================================

def foliation_score(
    eff_dims: np.ndarray,
    coherent_fraction: float,
    mean_ari: float,
    d_manifold: int,
) -> float:
    """Score composto F = (1 - H/Hmax) * coherence * ARI.

    Args:
        eff_dims: [K] dimensoes efectivas.
        coherent_fraction: Fracao de pares coerentes.
        mean_ari: ARI medio.
        d_manifold: Dimensao do manifold.

    Returns:
        F em [0, 1].
    """
    counts = np.bincount(eff_dims, minlength=d_manifold + 1)[1:]
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log(probs))
    H_max = np.log(d_manifold)

    F = (1.0 - H / H_max) * coherent_fraction * mean_ari
    return float(max(F, 0.0))


# ============================================================
# FASE 9: Coords Correlation (ANOVA)
# ============================================================

def coords_correlation(
    coords: np.ndarray,
    labels: np.ndarray,
    d_manifold: int,
) -> dict:
    """ANOVA F-stat por eixo do manifold vs cluster.

    Args:
        coords: [N, D] coordenadas.
        labels: [N] cluster labels.
        d_manifold: Dimensao do manifold.

    Returns:
        Dict com F-stat e p-value por eixo.
    """
    logger.info("[FASE 9] Coords Correlation (ANOVA)")

    n_cells = labels.max() + 1
    groups = [coords[labels == k] for k in range(n_cells) if (labels == k).sum() >= 10]

    anova = {}
    for ax in range(d_manifold):
        ax_groups = [g[:, ax] for g in groups if len(g) > 0]
        if len(ax_groups) < 2:
            continue
        F_stat, p_val = f_oneway(*ax_groups)
        anova[f"coord_{ax}"] = {
            "F": float(F_stat) if np.isfinite(F_stat) else 0.0,
            "p": float(p_val) if np.isfinite(p_val) else 1.0,
        }
        logger.info("  coord_%d: F=%.1f, p=%.2e", ax, F_stat, p_val)

    return anova


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Voronoi Foliation para DRM Transformer")
    parser.add_argument("--coords", required=True, help="Caminho para coords.npy")
    parser.add_argument("--G-diag", default=None, help="Caminho para G_diag.npy")
    parser.add_argument("--gamma", default=None, help="Caminho para gamma.npy")
    parser.add_argument("--mass", default=None, help="Caminho para mass.npy")
    parser.add_argument("--output-dir", default="eval_results/foliation")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--homology-points", type=int, default=1500)
    parser.add_argument("--use-gamma-distance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    t0 = time.time()

    # Carregar dados
    coords = np.load(args.coords).astype(np.float32)
    D = coords.shape[1]
    logger.info("[LOAD] coords: %s (d_manifold=%d)", coords.shape, D)

    G_diag = np.load(args.G_diag).astype(np.float32) if args.G_diag else None
    gamma_arr = np.load(args.gamma).astype(np.float32) if args.gamma else None
    mass_arr = np.load(args.mass).astype(np.float32) if args.mass else None

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # FASE 1
    vor = voronoi_tessellation(
        coords, n_seeds=args.n_seeds,
        G_diag=G_diag, gamma=gamma_arr,
        use_gamma_distance=args.use_gamma_distance,
    )
    labels = vor["labels"]
    centers = vor["centers"]

    # FASE 2
    ltsa = ltsa_per_cell(coords, labels)
    results["ltsa"] = {
        "mean_eff_dim": float(ltsa["eff_dims"].mean()),
        "eff_dims_dist": np.bincount(ltsa["eff_dims"], minlength=D + 1)[1:].tolist(),
        "cell_sizes": ltsa["cell_sizes"].tolist(),
    }

    # FASE 3
    coh = tangent_coherence(centers, ltsa["eigenvectors"], ltsa["eff_dims"])
    results["coherence"] = {
        "mean_angle": coh["mean_angle"],
        "median_angle": coh["median_angle"],
        "coherent_fraction": coh["coherent_fraction"],
    }

    # FASE 4: Reeb Graph
    reeb_results = {}

    if gamma_arr is not None:
        reeb_results["gamma"] = compute_reeb_graph(
            coords, labels, gamma_arr.ravel(), func_name="gamma",
        )

    if mass_arr is not None:
        reeb_results["mass"] = compute_reeb_graph(
            coords, labels, mass_arr.ravel(), func_name="mass",
        )

    # Tambem usar coord_0 como funcao de Morse
    reeb_results["coord_0"] = compute_reeb_graph(
        coords, labels, coords[:, 0], func_name="coord_0",
    )
    results["reeb"] = reeb_results

    # FASE 5
    homology = compute_homology(
        coords, G_diag=G_diag, max_points=args.homology_points,
    )
    results["homology"] = homology

    # FASE 6
    results["null_models"] = {
        "shuffled": null_model(coords, centers, "shuffled"),
        "uniform": null_model(coords, centers, "uniform"),
    }

    # FASE 7
    stab = stability_test(coords, n_seeds=args.n_seeds, n_restarts=args.n_restarts)
    results["stability"] = stab

    # FASE 8
    F = foliation_score(
        ltsa["eff_dims"], coh["coherent_fraction"],
        stab["mean_ari"], D,
    )
    results["foliation_score"] = F
    logger.info("[FASE 8] Foliation Score F = %.4f", F)

    # FASE 9
    anova = coords_correlation(coords, labels, D)
    results["coords_correlation"] = anova

    # Config
    results["config"] = {
        "n_seeds": args.n_seeds,
        "n_restarts": args.n_restarts,
        "homology_points": args.homology_points,
        "use_gamma_distance": args.use_gamma_distance,
        "n_vectors": len(coords),
        "d_manifold": D,
    }

    # Salvar
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "voronoi_labels.npy", labels)
    np.save(out_dir / "voronoi_centers.npy", centers)
    np.save(out_dir / "eigenvalues.npy", ltsa["eigenvalues"])
    np.save(out_dir / "eff_dims.npy", ltsa["eff_dims"])
    np.save(out_dir / "coherence_matrix.npy", coh["coherence_matrix"])

    with open(out_dir / "foliation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for key, reeb in reeb_results.items():
        with open(out_dir / f"reeb_{key}.json", "w") as f:
            json.dump(reeb, f, indent=2, default=str)

    if homology and not homology.get("skipped"):
        with open(out_dir / "homology.json", "w") as f:
            json.dump(homology, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("[DONE] %.1fs -> %s/", elapsed, out_dir)
    logger.info("  F=%.4f, topology=%s, ARI=%.4f",
                F, homology.get("topology", "?"), stab["mean_ari"])


if __name__ == "__main__":
    main()
