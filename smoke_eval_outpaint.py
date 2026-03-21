#!/usr/bin/env python3
"""
Quick sanity check for the outpaint eval path (condition -> sample_outpaint -> VAE decode -> HTML).
Does NOT run training. Use before long jobs to verify scheduler/VAE/dtype/inference.

Example:
  CUDA_VISIBLE_DEVICES=0 python smoke_eval_outpaint.py \\
    --pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev \\
    --init_lora_weights=Insta360-Research/DiT360-Panorama-Image-Generation \\
    --test_id_list=./data/panomtdu_test.json \\
    --pano_root=s3://your-bucket/pano \\
    --num_inference_steps=4 --num_samples=1

Default --inference_dtype=auto uses fp16 on CUDA, matching training eval under --precision=16-mixed.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from argparse import Namespace
from typing import Any, Dict, List

import torch
from peft import LoraConfig
from PIL import Image

from src.dit360_outpaint import DiT360Outpaint
from src.outpaint_dataset import RandomPerspOutpaintDataset
from src.outpaint_eval_utils import resolve_smoke_inference_dtype, run_one_outpaint_eval
from src.pipeline import DiT360Pipeline


def _norm_to_uint8_img(tensor: torch.Tensor):
    x = tensor.detach().cpu().clamp(-1.0, 1.0)
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")


def _save_eval_html(output_dir: str, rows: List[Dict[str, str]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "smoke_eval.html")
    parts = [
        "<html><head><meta charset='utf-8'><title>Outpaint smoke eval</title>",
        "<style>body{font-family:Arial,sans-serif;padding:20px;} table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ccc;padding:8px;vertical-align:top;} img{max-width:100%;height:auto;}</style>",
        "</head><body><h2>Outpaint smoke eval</h2>",
        "<table><tr><th>Sample</th><th>Input</th><th>Text</th><th>Generated</th><th>Target</th></tr>",
    ]
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row['sample_id']}</td>"
            f"<td><img src='{row['input']}'></td>"
            f"<td>{row['text']}</td>"
            f"<td><img src='{row['generated']}'></td>"
            f"<td><img src='{row['target']}'></td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return html_path


def _model_args_from_cli(a: argparse.Namespace) -> Namespace:
    """Fields required by DiT360Outpaint + configure_optimizers (unused here but saved in hparams)."""
    return Namespace(
        pretrained_model_name_or_path=a.pretrained_model_name_or_path,
        init_lora_weights=a.init_lora_weights,
        padding_n=a.padding_n,
        guidance_scale=a.guidance_scale,
        perspective_size=a.perspective_size,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        learning_rate=2e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        adam_weight_decay=1e-2,
        max_steps=1,
        warmup_steps=1,
        lr_scheduler="cosine",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Smoke test outpaint eval without training.")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--init_lora_weights", type=str, default="Insta360-Research/DiT360-Panorama-Image-Generation")
    p.add_argument("--test_id_list", type=str, required=True)
    p.add_argument("--pano_root", type=str, required=True)
    p.add_argument("--caption_map_file", type=str, default=None)
    p.add_argument("--caption_template", type=str, default="This is a panorama image.")
    p.add_argument("--image_ext", type=str, default="jpg")
    p.add_argument("--exclude_prefixes", type=str, nargs="+", default=["DiT360_"])
    p.add_argument("--pano_height", type=int, default=512)
    p.add_argument("--pano_width", type=int, default=1024)
    p.add_argument("--perspective_size", type=int, default=512)
    p.add_argument("--min_views", type=int, default=1)
    p.add_argument("--max_views", type=int, default=10)
    p.add_argument("--min_fov", type=float, default=75.0)
    p.add_argument("--max_fov", type=float, default=105.0)
    p.add_argument("--min_pitch", type=float, default=-30.0)
    p.add_argument("--max_pitch", type=float, default=30.0)
    p.add_argument("--padding_n", type=int, default=1)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--gaussian_init_lora", action="store_true")
    p.add_argument("--lora_drop_out", type=float, default=0.05)
    p.add_argument("--num_samples", type=int, default=1, help="How many test IDs to run.")
    p.add_argument("--num_inference_steps", type=int, default=4, help="Keep small for speed (e.g. 4–8).")
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--output_dir", type=str, default="outpaint_eval_smoke")
    p.add_argument("--ckpt", type=str, default=None, help="Optional Lightning .ckpt to load state_dict.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--inference_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="Must match training eval: use 'auto' (fp16 on CUDA, same as 16-mixed) or match --precision.",
    )
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[smoke] CUDA not available; running on CPU will be very slow.")

    text_pipe = DiT360Pipeline.from_pretrained(a.pretrained_model_name_or_path, vae=None, transformer=None)

    ds = RandomPerspOutpaintDataset(
        id_list_file=a.test_id_list,
        pano_root=a.pano_root,
        pano_height=a.pano_height,
        pano_width=a.pano_width,
        image_ext=a.image_ext,
        caption_map_file=a.caption_map_file,
        caption_template=a.caption_template,
        exclude_prefixes=a.exclude_prefixes,
        min_views=a.min_views,
        max_views=a.max_views,
        min_fov=a.min_fov,
        max_fov=a.max_fov,
        min_pitch=a.min_pitch,
        max_pitch=a.max_pitch,
        perspective_size=a.perspective_size,
    )

    model_args = _model_args_from_cli(a)
    lora_config = LoraConfig(
        r=a.rank,
        lora_alpha=a.lora_alpha,
        init_lora_weights="gaussian" if a.gaussian_init_lora else True,
        target_modules=["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0"],
        bias=a.lora_bias,
        lora_dropout=a.lora_drop_out,
    )
    model = DiT360Outpaint(model_args, lora_config=lora_config)
    model = model.to(device)
    model.on_fit_start()
    model.eval()

    if a.ckpt:
        ckpt: Dict[str, Any] = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        print(f"[smoke] Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")

    os.makedirs(a.output_dir, exist_ok=True)
    total = len(ds)
    k = min(a.num_samples, total)
    subset_rng = random.Random(a.eval_seed)
    eval_indices = subset_rng.sample(range(total), k=k)
    print(f"[smoke] Subset: {k}/{total} indices (random.sample, eval_seed={a.eval_seed})")

    rng_state = random.getstate()
    random.seed(a.eval_seed)

    inference_dtype = resolve_smoke_inference_dtype(device, a.inference_dtype)
    print(f"[smoke] inference_dtype={inference_dtype} (training eval uses pl_module.dtype, typically float16 with 16-mixed)")

    rows = []
    for i, idx in enumerate(eval_indices):
        sample = ds[idx]
        with torch.no_grad():
            condition, generated, target_cpu = run_one_outpaint_eval(
                model,
                sample,
                text_pipe,
                perspective_size=a.perspective_size,
                num_inference_steps=a.num_inference_steps,
                inference_seed=a.eval_seed + i,
                inference_dtype=inference_dtype,
            )

        sid = ds.ids[idx]
        in_name = f"{i:03d}_{sid}_input.png"
        gen_name = f"{i:03d}_{sid}_generated.png"
        tgt_name = f"{i:03d}_{sid}_target.png"
        Image.fromarray(_norm_to_uint8_img(condition.cpu())).save(os.path.join(a.output_dir, in_name))
        Image.fromarray(_norm_to_uint8_img(generated.cpu())).save(os.path.join(a.output_dir, gen_name))
        Image.fromarray(_norm_to_uint8_img(target_cpu)).save(os.path.join(a.output_dir, tgt_name))
        rows.append(
            {"sample_id": sid, "input": in_name, "text": sample.captions, "generated": gen_name, "target": tgt_name}
        )
        print(f"[smoke] OK sample {i}: {sid}")

    random.setstate(rng_state)
    html = _save_eval_html(a.output_dir, rows)
    print(f"[smoke] Done. Open: {html}")


if __name__ == "__main__":
    main()
