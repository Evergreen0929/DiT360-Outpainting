"""
Shared outpaint eval path used by training (PeriodicOutpaintEvalCallback) and smoke_eval_outpaint.py.
Keeps inference logic identical; pass the same inference_dtype as training (pl_module.dtype under 16-mixed).
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
    feather_sigma: float = 8.0,
    feather_kernel: Optional[int] = None,
) -> torch.Tensor:
    """
    Eval-only RGB blend in [-1, 1]: keep known (edited-out) pixels from `condition`, rest from `generated`.
    unknown_mask: [1, H, W], 1 = use generated, 0 = use condition. Gaussian feather widens the transition.
    feather_sigma <= 0: hard binary blend (no blur).
    """
    g = generated.float()
    c = condition.float()
    u = unknown_mask.float().clamp(0.0, 1.0)
    if u.dim() == 2:
        u = u.unsqueeze(0)
    if feather_sigma > 0:
        u = _feather_mask_gaussian(u.unsqueeze(0), sigma=feather_sigma, kernel_size=feather_kernel).squeeze(0)
    u3 = u.expand_as(g)
    out = g * u3 + c * (1.0 - u3)
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
    eval_feather_sigma: float = 8.0,
    eval_feather_kernel: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        condition: [3,H,W] on device (unchanged hard-edge input)
        generated: [3,H,W] on device — after optional RGB composite (known region from condition)
        target_pixels: [3,H,W] CPU float (dataset tensor, for saving)

    The mask passed to ``sample_outpaint`` is dilated inside the Lightning module (if
    ``outpaint_mask_dilate_px`` > 0); ``composite_generated_with_condition`` uses the
    original undilated ``unknown`` from ``build_condition_from_target`` so pasted known
    pixels align with the true FOV boundary.
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

    generated = pl_module.sample_outpaint(
        condition_pixels=condition.unsqueeze(0).to(device=device, dtype=inference_dtype),
        unknown_masks=unknown.unsqueeze(0).to(device=device, dtype=inference_dtype),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids,
        num_inference_steps=num_inference_steps,
        seed=inference_seed,
        inference_dtype=inference_dtype,
        guidance_scale_override=OUTPAINT_EVAL_GUIDANCE_SCALE,
    )[0]

    generated = composite_generated_with_condition(
        generated,
        condition,
        unknown,
        feather_sigma=eval_feather_sigma,
        feather_kernel=eval_feather_kernel,
    )

    return condition, generated, sample.target_pixel_values


def resolve_smoke_inference_dtype(device: torch.device, mode: str) -> torch.dtype:
    """mode: 'auto' (fp16 on cuda), 'fp16', 'fp32'."""
    if mode == "fp16":
        return torch.float16
    if mode == "fp32":
        return torch.float32
    # auto
    return torch.float16 if device.type == "cuda" else torch.float32
