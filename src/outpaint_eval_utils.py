"""
Shared outpaint eval path used by training (PeriodicOutpaintEvalCallback) and smoke_eval_outpaint.py.
Keeps inference logic identical; pass the same inference_dtype as training (pl_module.dtype under mixed precision).
Returns condition, generated_raw (decode only), generated (after RGB composite), target.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.outpaint_dataset import OutpaintSample, build_condition_from_target

# DiT360-style eval: training uses guidance_scale=1.0; paper uses ~3.0 for inference / visual eval.
OUTPAINT_EVAL_GUIDANCE_SCALE = 3.0


def _gaussian_blur2d_1ch(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for [B,1,H,W] float masks / weights."""
    if kernel_size < 3:
        return x
    device, dtype = x.device, x.dtype
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    g_h = g.view(1, 1, 1, kernel_size)
    g_v = g.view(1, 1, kernel_size, 1)
    c = kernel_size // 2
    y = F.conv2d(x, g_v.expand(x.size(1), 1, kernel_size, 1), padding=(c, 0), groups=x.size(1))
    y = F.conv2d(y, g_h.expand(y.size(1), 1, 1, kernel_size), padding=(0, c), groups=y.size(1))
    return y


def _feather_mask_gaussian(
    mask_b1hw: torch.Tensor,
    *,
    sigma: float,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    if sigma <= 0:
        return mask_b1hw
    if kernel_size is None:
        k = 2 * int(math.ceil(3.0 * sigma)) + 1
        kernel_size = max(3, k | 1)
    elif kernel_size % 2 == 0:
        kernel_size += 1
    blurred = _gaussian_blur2d_1ch(mask_b1hw, kernel_size, sigma)
    return blurred.clamp(0.0, 1.0)


def composite_generated_with_condition(
    generated: torch.Tensor,
    condition: torch.Tensor,
    unknown_mask: torch.Tensor,
    *,
    feather_sigma: float = 32.0,
    feather_kernel: Optional[int] = None,
) -> torch.Tensor:
    """
    Eval-only RGB blend in [-1, 1]: known pixels from `condition`, inpaint from `generated`.

    unknown_mask: [1, H, W], 1 = inpaint (use generated), 0 = known (use condition).

    Gaussian feather is applied only in the sense of softening the **known→generated** boundary:
    blurring the mask alone would pull weights below 1 inside the hole and mix in the gray
    `condition` there; instead we use ``alpha = max(u_hard, u_blurred)`` so any pixel that
    was inpaint in the hard mask stays 100% ``generated``, while known pixels near the edge
    can still pick up a smooth transition via the blurred field.
    """
    g = generated.float()
    c = condition.float()
    u_hard = unknown_mask.float().clamp(0.0, 1.0)
    if u_hard.dim() == 2:
        u_hard = u_hard.unsqueeze(0)
    if feather_sigma > 0:
        u_soft = _feather_mask_gaussian(
            u_hard.unsqueeze(0), sigma=feather_sigma, kernel_size=feather_kernel
        ).squeeze(0)
        alpha = torch.maximum(u_hard, u_soft)
    else:
        alpha = u_hard
    alpha = alpha.clamp(0.0, 1.0)
    a3 = alpha.expand_as(g)
    out = g * a3 + c * (1.0 - a3)
    return out.to(dtype=generated.dtype)


@torch.no_grad()
def run_one_outpaint_eval(
    pl_module,
    sample: OutpaintSample,
    text_encoding_pipeline,
    *,
    perspective_size: int,
    num_inference_steps: int,
    inference_seed: int,
    inference_dtype: torch.dtype,
    eval_feather_sigma: float = 32.0,
    eval_feather_kernel: Optional[int] = None,
    inference_valid_mask_blur_kernel_px: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        condition: [3,H,W] on device (unchanged hard-edge input)
        generated_raw: [3,H,W] on device — VAE decode of sample_outpaint only (no RGB paste-back)
        generated: [3,H,W] on device — after optional RGB composite (known region from condition)
        target_pixels: [3,H,W] CPU float (dataset tensor, for saving)

    The mask passed to ``sample_outpaint`` is optionally dilated in the Lightning module (if
    ``outpaint_mask_dilate_px`` > 0). If ``inference_valid_mask_blur_kernel_px`` >= 3, the valid
    (known) mask is Gaussian-blurred in pixel space before latent resize (inference-only; training
    loss path does not use this). Then nearest resize to latent size.
    ``composite_generated_with_condition`` uses the original undilated ``unknown`` from
    ``build_condition_from_target`` so pasted known pixels align with the true FOV boundary.
    """
    device = pl_module.device
    target_pixels = sample.target_pixel_values.to(device=device, dtype=inference_dtype)
    view_params = sample.view_params.to(device=device)
    condition, unknown = build_condition_from_target(
        target_pixel_values=target_pixels,
        view_params=view_params,
        num_views=sample.num_views,
        perspective_size=perspective_size,
    )

    prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
        [sample.captions], prompt_2=None
    )
    prompt_embeds = prompt_embeds.to(device=device, dtype=inference_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=inference_dtype)
    text_ids = text_ids.to(device=device, dtype=inference_dtype)

    generated_raw = pl_module.sample_outpaint(
        condition_pixels=condition.unsqueeze(0).to(device=device, dtype=inference_dtype),
        unknown_masks=unknown.unsqueeze(0).to(device=device, dtype=inference_dtype),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        num_inference_steps=num_inference_steps,
        seed=inference_seed,
        inference_dtype=inference_dtype,
        guidance_scale_override=OUTPAINT_EVAL_GUIDANCE_SCALE,
        valid_mask_blur_kernel_px=inference_valid_mask_blur_kernel_px,
    )[0]

    generated = composite_generated_with_condition(
        generated_raw,
        condition,
        unknown,
        feather_sigma=eval_feather_sigma,
        feather_kernel=eval_feather_kernel,
    )

    return condition, generated_raw, generated, sample.target_pixel_values


def resolve_smoke_inference_dtype(
    device: torch.device,
    mode: str,
    *,
    lightning_precision: Optional[str] = None,
) -> torch.dtype:
    """
    mode: auto | fp16 | fp32 | bf16.
    When mode is auto and lightning_precision is set, match Lightning training eval (pl_module.dtype):
    bf16-mixed -> bfloat16 on CUDA; 16-mixed -> float16 on CUDA; 32-* -> float32.
    """
    if mode == "fp16":
        return torch.float16
    if mode == "fp32":
        return torch.float32
    if mode == "bf16":
        return torch.bfloat16
    if mode != "auto":
        raise ValueError(f"Unknown inference_dtype mode: {mode!r}")
    if device.type == "cuda" and lightning_precision:
        lp = str(lightning_precision).lower().replace(" ", "").replace("_", "-")
        if "bf16" in lp:
            return torch.bfloat16
        if lp.startswith("32") or lp.startswith("64"):
            return torch.float32
        if "16" in lp:
            return torch.float16
    return torch.float16 if device.type == "cuda" else torch.float32
