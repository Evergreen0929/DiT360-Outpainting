#!/usr/bin/env python3
"""
Monte Carlo estimate of P(no outpaint supervision) under the same view sampling as
RandomPerspOutpaintDataset._sample_view_params + build_condition_from_target.

"No supervision" means unknown_mask.sum() == 0 (ERP fully covered by projected known regions),
matching the case where training loss is exactly 0 with the current masked loss.

Usage (defaults match train_outpaint_lora.sh pano / view hyperparams):
  python scripts/estimate_outpaint_no_supervision_rate.py --trials 50000 --seed 0
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.outpaint_dataset import build_condition_from_target


def sample_view_params(
    min_views: int,
    max_views: int,
    min_fov: float,
    max_fov: float,
    min_pitch: float,
    max_pitch: float,
) -> tuple[torch.Tensor, int]:
    """Match RandomPerspOutpaintDataset._sample_view_params."""
    candidates = list(range(min_views, max_views + 1))
    weights = [max_views + 1 - k for k in candidates]
    n_views = random.choices(candidates, weights=weights, k=1)[0]
    params = torch.zeros((max_views, 3), dtype=torch.float32)
    for i in range(n_views):
        params[i, 0] = random.uniform(-180.0, 180.0)
        params[i, 1] = random.uniform(min_pitch, max_pitch)
        params[i, 2] = random.uniform(min_fov, max_fov)
    return params, n_views


def n_views_distribution(min_views: int, max_views: int) -> dict[int, float]:
    candidates = list(range(min_views, max_views + 1))
    weights = [max_views + 1 - k for k in candidates]
    s = float(sum(weights))
    return {k: w / s for k, w in zip(candidates, weights)}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=50_000, help="Monte Carlo samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pano_height", type=int, default=512)
    p.add_argument("--pano_width", type=int, default=1024)
    p.add_argument("--perspective_size", type=int, default=512)
    p.add_argument("--min_views", type=int, default=1)
    p.add_argument("--max_views", type=int, default=10)
    p.add_argument("--min_fov", type=float, default=75.0)
    p.add_argument("--max_fov", type=float, default=105.0)
    p.add_argument("--min_pitch", type=float, default=-30.0)
    p.add_argument("--max_pitch", type=float, default=30.0)
    p.add_argument(
        "--latent_downsample",
        type=int,
        default=8,
        help="If >0, also require unknown.sum()==0 after nearest downsample (typical VAE 8x). 0=skip.",
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    dummy_target = torch.zeros(3, args.pano_height, args.pano_width, device=device, dtype=torch.float32)

    no_sup_pixel = 0
    no_sup_latent = 0
    by_k_counts: dict[int, int] = defaultdict(int)
    by_k_no_sup: dict[int, int] = defaultdict(int)

    for _ in range(args.trials):
        view_params, n_views = sample_view_params(
            args.min_views,
            args.max_views,
            args.min_fov,
            args.max_fov,
            args.min_pitch,
            args.max_pitch,
        )
        _, unknown = build_condition_from_target(
            target_pixel_values=dummy_target,
            view_params=view_params,
            num_views=n_views,
            perspective_size=args.perspective_size,
        )
        # unknown: [1, H, W]
        u_sum = float(unknown.sum().item())
        by_k_counts[n_views] += 1
        if u_sum == 0.0:
            no_sup_pixel += 1
            by_k_no_sup[n_views] += 1

        if args.latent_downsample > 0:
            uh, uw = args.pano_height // args.latent_downsample, args.pano_width // args.latent_downsample
            if uh < 1 or uw < 1:
                raise SystemExit("latent_downsample too large for pano size")
            u4 = unknown.unsqueeze(0)
            u_lat = F.interpolate(u4, size=(uh, uw), mode="nearest")
            if float(u_lat.sum().item()) == 0.0:
                no_sup_latent += 1

    p_all = no_sup_pixel / args.trials
    lo, hi = wilson_ci(no_sup_pixel, args.trials)

    print("=== Outpaint \"no supervision\" (unknown mask all zero) ===")
    print(f"trials={args.trials}, seed={args.seed}")
    print(
        f"pano={args.pano_width}x{args.pano_height}, perspective_size={args.perspective_size}, "
        f"views=[{args.min_views},{args.max_views}], fov=[{args.min_fov},{args.max_fov}], "
        f"pitch=[{args.min_pitch},{args.max_pitch}]"
    )
    print()
    print("P(n_views=k) analytic (dataset sampling):")
    dist = n_views_distribution(args.min_views, args.max_views)
    for k in sorted(dist):
        print(f"  k={k:2d}: {dist[k]:.4f}")
    print()
    print(f"P(no supervision | pixel ERP mask)  ≈ {p_all:.6f}  ({no_sup_pixel}/{args.trials})")
    print(f"  95% Wilson CI: [{lo:.6f}, {hi:.6f}]")
    if args.latent_downsample > 0:
        p_lat = no_sup_latent / args.trials
        lo2, hi2 = wilson_ci(no_sup_latent, args.trials)
        print(
            f"P(no supervision | after {args.latent_downsample}x nearest downsample) "
            f"≈ {p_lat:.6f}  ({no_sup_latent}/{args.trials})"
        )
        print(f"  95% Wilson CI: [{lo2:.6f}, {hi2:.6f}]")
    print()
    print("Empirical P(no supervision | n_views=k) (Monte Carlo, may be noisy for rare k):")
    for k in sorted(by_k_counts.keys()):
        c = by_k_counts[k]
        ns = by_k_no_sup[k]
        print(f"  k={k:2d}: {ns}/{c} = {ns/c:.6f}")


if __name__ == "__main__":
    main()
