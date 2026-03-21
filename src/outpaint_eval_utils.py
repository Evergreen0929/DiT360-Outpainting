"""
Shared outpaint eval path used by training (PeriodicOutpaintEvalCallback) and smoke_eval_outpaint.py.
Keeps inference logic identical; pass the same inference_dtype as training (pl_module.dtype under 16-mixed).
"""

from __future__ import annotations

import torch

from src.outpaint_dataset import OutpaintSample, build_condition_from_target

# DiT360-style eval: training uses guidance_scale=1.0; paper uses ~3.0 for inference / visual eval.
OUTPAINT_EVAL_GUIDANCE_SCALE = 3.0


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        condition: [3,H,W] on device
        generated: [3,H,W] on device
        target_pixels: [3,H,W] CPU float (dataset tensor, for saving)
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

    return condition, generated, sample.target_pixel_values


def resolve_smoke_inference_dtype(device: torch.device, mode: str) -> torch.dtype:
    """mode: 'auto' (fp16 on cuda), 'fp16', 'fp32'."""
    if mode == "fp16":
        return torch.float16
    if mode == "fp32":
        return torch.float32
    # auto
    return torch.float16 if device.type == "cuda" else torch.float32
