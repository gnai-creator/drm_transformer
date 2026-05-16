# Voronoi/Foliation Findings

## Problem

The current baseline checkpoint produced a degenerate manifold cloud:

- `coords std ~= [0.0028, 0.0057, 0.0066, 0.0038]`
- `gamma std ~= 0.0013`
- persistent homology reported `H1=11`, `H2=3`

Because the coordinates are concentrated near `0.5` on every axis, the
homology counts are likely dominated by subsampling and threshold artifacts
rather than a stable toroidal geometry.

## Fixes Implemented

1. Robust homology evaluation:
   - z-score normalization before ripser;
   - multiple homology subsamples;
   - saved first-restart persistence diagrams;
   - `t2_stable_fraction` for stable `H1=2`, `H2=1` validation.

2. Manifold anti-collapse training loss:
   - `manifold_variance_loss` encourages coordinate usage;
   - gradients now flow into `q_to_manifold` for geometry losses.

3. Toroidal regularization:
   - `torus_regularization_loss` treats the first four manifold dimensions as
     two circular pairs;
   - configs expose `lambda_torus`, `torus_target_radius`, and
     `torus_coverage_weight`.

## Next Validation

Retrain `configs/baselines/small_3.5M.yaml`, extract DRM vectors, then run:

```bash
python scripts/voronoi_foliation_drm.py \
    --coords eval-results/foliation_3.5m/drm_coords.npy \
    --G-diag eval-results/foliation_3.5m/drm_G_diag.npy \
    --gamma eval-results/foliation_3.5m/drm_gamma.npy \
    --output-dir eval-results/foliation_3.5m \
    --n-seeds 30 \
    --homology-points 800 \
    --homology-restarts 5
```

The torus target is considered validated only if `H1=2`, `H2=1` is stable
across subsamples (`t2_stable_fraction >= 0.6`).
