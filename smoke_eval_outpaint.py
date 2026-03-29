#!/usr/bin/env python3
"""
Smoke / offline eval: same path as training PeriodicOutpaintEvalCallback (run_one_outpaint_eval).

Loads the latest step checkpoint under --training_save_dir unless --ckpt is set.
Saves condition input, raw VAE decode (before RGB paste-back), composited result, and GT.

Example:
  CUDA_VISIBLE_DEVICES=0 python smoke_eval_outpaint.py \\
    --pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev \\
    --init_lora_weights=Insta360-Research/DiT360-Panorama-Image-Generation \\
    --training_save_dir=outpaint_model_saved --training_tb_version=3 \\
    --test_id_list=./data/panomtdu_test.json \\
    --pano_root=s3://your-bucket/pano \\
    --precision=bf16-mixed --num_inference_steps=30

Use the same --precision, LoRA hparams, geometry, mask dilation, and eval feather flags as training.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig
from PIL import Image

from src.dit360_outpaint import DiT360Outpaint
from src.outpaint_dataset import RandomPerspOutpaintDataset, build_condition_from_target
from src.outpaint_eval_utils import resolve_smoke_inference_dtype, run_one_outpaint_eval
from src.pipeline import DiT360Pipeline


def _norm_to_uint8_img(tensor: torch.Tensor):
    x = tensor.detach().cpu().clamp(-1.0, 1.0)
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")


def resolve_training_checkpoint_file(path: str) -> str:
    """
    Accept a single Lightning .ckpt file, or a DeepSpeed checkpoint directory (name often ends in .ckpt).
    Resolves to the file that torch.load can read: …/checkpoint/mp_rank_00_model_states.pt
    """
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return str(p)
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    inner = p / "checkpoint" / "mp_rank_00_model_states.pt"
    if inner.is_file():
        return str(inner)
    inner = p / "mp_rank_00_model_states.pt"
    if inner.is_file():
        return str(inner)
    raise FileNotFoundError(
        f"Expected DeepSpeed layout under {path}: checkpoint/mp_rank_00_model_states.pt (or mp_rank_00_model_states.pt)."
    )


def _step_from_outpaint_ckpt_name(name: str) -> Optional[int]:
    """Parse training step from dir/file names like outpaint_step_00004000.ckpt or outpaint_step_step=00004000.ckpt."""
    m = re.search(r"(\d+)\.ckpt$", name)
    return int(m.group(1)) if m else None


def find_latest_outpaint_checkpoint(save_dir: str) -> Optional[str]:
    """
    Prefer highest-step DeepSpeed folder outpaint_step*.ckpt (directory) or single-file step ckpt;
    else newest last.ckpt; else newest *.ckpt file.
    Returns a path passable to resolve_training_checkpoint_file (dir or file).
    """
    root = Path(save_dir)
    if not root.is_dir():
        return None
    step_dirs: List[tuple[int, Path]] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if not (p.name.startswith("outpaint_step") and p.name.endswith(".ckpt")):
            continue
        s = _step_from_outpaint_ckpt_name(p.name)
        if s is not None:
            step_dirs.append((s, p))
    if step_dirs:
        best = max(step_dirs, key=lambda x: x[0])[1]
        return str(best)

    step_files = [
        p
        for p in root.rglob("outpaint_step_*.ckpt")
        if p.is_file() and _step_from_outpaint_ckpt_name(p.name) is not None
    ]
    if step_files:
        best = max(step_files, key=lambda p: _step_from_outpaint_ckpt_name(p.name) or -1)
        return str(best)

    last_ckpts = list(root.rglob("last.ckpt"))
    if last_ckpts:
        return str(max(last_ckpts, key=lambda p: p.stat().st_mtime))
    all_ckpt = [p for p in root.rglob("*.ckpt") if p.is_file()]
    if not all_ckpt:
        return None
    return str(max(all_ckpt, key=lambda p: p.stat().st_mtime))


def _tensor_entries(d: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if isinstance(v, torch.Tensor)}


def _state_dict_from_loaded_checkpoint(obj: Any, *, verbose: bool = False) -> Dict[str, Any]:
    """
    Lightning+DeepSpeed often stores weights under ``module``, not top-level ``state_dict``.
    Using ``obj.get('state_dict', obj)`` on the full pickle would skip real weights and fail to
    load trained LoRA (symptoms can look like wrong inpainting regions).
    """
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint root must be dict, got {type(obj)}")

    candidates: List[tuple[str, dict]] = []
    for key in ("state_dict", "module"):
        inner = obj.get(key)
        if isinstance(inner, dict):
            candidates.append((key, inner))
    tensor_counts = [(name, len(_tensor_entries(d))) for name, d in candidates]
    if verbose:
        print(f"[smoke][ckpt] tensor counts by section: {tensor_counts}")

    raw: Optional[dict] = None
    best_n = -1
    for _name, d in candidates:
        n = len(_tensor_entries(d))
        if n > best_n:
            best_n = n
            raw = d

    if raw is None or best_n < 8:
        raw = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        best_n = len(raw)
        if verbose:
            print(f"[smoke][ckpt] fallback to top-level tensors only: {best_n} entries")

    if best_n == 0:
        raise ValueError(
            "No tensor weights in checkpoint (expected non-empty 'module' or 'state_dict'). "
            "For DeepSpeed, use …/checkpoint/mp_rank_00_model_states.pt."
        )

    sd = _tensor_entries(raw)
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = k
        while nk.startswith("module."):
            nk = nk[len("module.") :]
        out[nk] = v
    if verbose and out:
        sk = sorted(out.keys())
        print(f"[smoke][ckpt] sample keys after strip: {sk[:5]} … ({len(out)} tensors)")
    return out


def _save_eval_html(output_dir: str, rows: List[Dict[str, str]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "smoke_eval.html")
    parts = [
        "<html><head><meta charset='utf-8'><title>Outpaint smoke eval</title>",
        "<style>body{font-family:Arial,sans-serif;padding:20px;} table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ccc;padding:8px;vertical-align:top;} img{max-width:100%;height:auto;}</style>",
        "</head><body><h2>Outpaint smoke eval</h2>",
        "<table><tr><th>Sample</th><th>Condition</th><th>Raw decode</th><th>Composited</th><th>GT</th><th>Text</th></tr>",
    ]
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row['sample_id']}</td>"
            f"<td><img src='{row['input']}'></td>"
            f"<td><img src='{row['generated_raw']}'></td>"
            f"<td><img src='{row['generated']}'></td>"
            f"<td><img src='{row['target']}'></td>"
            f"<td>{row['text']}</td>"
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
        outpaint_mask_dilate_px=a.outpaint_mask_dilate_px,
        precision=a.precision,
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
    p = argparse.ArgumentParser(description="Smoke test outpaint eval without training; align with train eval.")
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
    p.add_argument(
        "--outpaint_mask_dilate_px",
        type=int,
        default=16,
        help="Same as training: dilate unknown mask in pixel space before latent resize; 0=off.",
    )
    p.add_argument("--min_views", type=int, default=1)
    p.add_argument("--max_views", type=int, default=10)
    p.add_argument("--min_fov", type=float, default=75.0)
    p.add_argument("--max_fov", type=float, default=105.0)
    p.add_argument("--min_pitch", type=float, default=-75.0)
    p.add_argument("--max_pitch", type=float, default=75.0)
    p.add_argument("--padding_n", type=int, default=1)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Lightning --precision from training; used for pl_module hparams and inference_dtype when auto.",
    )
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--gaussian_init_lora", action="store_true")
    p.add_argument("--lora_drop_out", type=float, default=0.05)
    p.add_argument("--num_samples", type=int, default=1, help="How many test IDs to run (same as eval_num_samples).")
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Same as training --eval_inference_steps.",
    )
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--output_dir", type=str, default="outpaint_eval_smoke")
    p.add_argument(
        "--training_save_dir",
        type=str,
        default=None,
        help="Trainer default_root_dir (--save_dir). Latest step under tb_logs (see --training_tb_version) unless --ckpt is set.",
    )
    p.add_argument(
        "--training_tb_version",
        type=int,
        default=None,
        help=(
            "Only when using --training_save_dir: restrict search to "
            "<save_dir>/tb_logs/version_<N>/ (e.g. 3 for version_3). "
            "If omitted, all versions are searched and the highest step wins (can pick an older run)."
        ),
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Checkpoint: single Lightning .ckpt, or DeepSpeed folder "
            "(…/outpaint_step_step=00004000.ckpt or …/checkpoint). "
            "Resolves to checkpoint/mp_rank_00_model_states.pt when needed."
        ),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--inference_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "fp32", "bf16"],
        help="auto: derive from --precision on CUDA (matches training eval pl_module.dtype).",
    )
    p.add_argument(
        "--eval_feather_sigma",
        type=float,
        default=32.0,
        help="Same as training --eval_feather_sigma.",
    )
    p.add_argument(
        "--eval_feather_kernel",
        type=int,
        default=None,
        help="Same as training --eval_feather_kernel (odd kernel; default from sigma).",
    )
    p.add_argument(
        "--eval_valid_mask_blur_kernel_px",
        type=int,
        default=32,
        help="Same as training --eval_valid_mask_blur_kernel_px; <3 disables.",
    )
    p.add_argument(
        "--verbose_ckpt",
        action="store_true",
        help="Print how many tensors were found under state_dict vs module (DeepSpeed debug).",
    )
    p.add_argument(
        "--debug_masks",
        action="store_true",
        help="Save *_unknown_inpaint.png: white=inpaint (unknown=1), black=keep — same mask as sample_outpaint uses before dilate.",
    )
    return p.parse_args()


def main():
    a = parse_args()
    if a.training_tb_version is not None and not a.training_save_dir and not a.ckpt:
        print("[smoke] --training_tb_version requires --training_save_dir (or use --ckpt with a full path).", file=sys.stderr)
        sys.exit(1)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[smoke] CUDA not available; running on CPU will be very slow.")

    ckpt_path: Optional[str] = None
    ckpt_file: Optional[str] = None
    if a.ckpt:
        ckpt_path = os.path.abspath(a.ckpt)
    elif a.training_save_dir:
        if a.training_tb_version is not None:
            vdir = (
                Path(a.training_save_dir).expanduser().resolve()
                / "tb_logs"
                / f"version_{a.training_tb_version}"
            )
            if not vdir.is_dir():
                print(
                    f"[smoke] No such logger version directory: {vdir} "
                    f"(set --training_tb_version to match tb_logs under your --training_save_dir).",
                    file=sys.stderr,
                )
                sys.exit(1)
            search_root = str(vdir)
            print(f"[smoke] Checkpoint search limited to: {search_root}")
        else:
            search_root = str(Path(a.training_save_dir).expanduser().resolve())
        found = find_latest_outpaint_checkpoint(search_root)
        if not found:
            hint = ""
            if a.training_tb_version is None:
                hint = " Pass --training_tb_version=N to use tb_logs/version_N only."
            print(
                f"[smoke] No checkpoint under {search_root!r} (expected **/outpaint_step*.ckpt/ or **/last.ckpt).{hint}",
                file=sys.stderr,
            )
            sys.exit(1)
        ckpt_path = found
        print(f"[smoke] Using latest checkpoint: {ckpt_path}")
    else:
        print("[smoke] No --ckpt or --training_save_dir: running init LoRA only (no trained weights).")

    if ckpt_path:
        try:
            ckpt_file = resolve_training_checkpoint_file(ckpt_path)
        except FileNotFoundError as e:
            print(f"[smoke] {e}", file=sys.stderr)
            sys.exit(1)
        if ckpt_file != ckpt_path:
            print(f"[smoke] Resolved load file: {ckpt_file}")

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

    inference_dtype = resolve_smoke_inference_dtype(
        device, a.inference_dtype, lightning_precision=a.precision if a.inference_dtype == "auto" else None
    )
    print(f"[smoke] inference_dtype={inference_dtype} (precision={a.precision!r}, inference_dtype mode={a.inference_dtype!r})")

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
    model = model.to(device, dtype=inference_dtype)
    model.on_fit_start()
    model.eval()

    if ckpt_file:
        load_kw: Dict[str, Any] = {"map_location": "cpu", "weights_only": False}
        try:
            load_kw["mmap"] = True
            ckpt_obj: Dict[str, Any] = torch.load(ckpt_file, **load_kw)
        except TypeError:
            load_kw.pop("mmap", None)
            ckpt_obj = torch.load(ckpt_file, **load_kw)
        sd = _state_dict_from_loaded_checkpoint(ckpt_obj, verbose=a.verbose_ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        ms = model.state_dict()
        lora_keys = [k for k in ms if "lora_" in k]
        missing_lora = [k for k in missing if "lora_" in k]
        print(f"[smoke] Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
        if lora_keys:
            pct = 100.0 * (len(lora_keys) - len(missing_lora)) / max(len(lora_keys), 1)
            print(f"[smoke] LoRA tensors: {len(lora_keys) - len(missing_lora)}/{len(lora_keys)} loaded ({pct:.1f}%).")
            if missing_lora:
                print(f"[smoke] WARNING: LoRA not loaded from ckpt (first keys): {missing_lora[:6]}")
                print(
                    "[smoke] Hint: DeepSpeed checkpoints store weights in ['module']; "
                    "this script now prefers that block — if you still see this, check key prefixes vs DiT360Outpaint.",
                    file=sys.stderr,
                )

    os.makedirs(a.output_dir, exist_ok=True)
    total = len(ds)
    k = min(a.num_samples, total)
    subset_rng = random.Random(a.eval_seed)
    eval_indices = subset_rng.sample(range(total), k=k)
    print(f"[smoke] Subset: {k}/{total} indices (random.sample, eval_seed={a.eval_seed})")

    rng_state = random.getstate()
    random.seed(a.eval_seed)

    rows = []
    for i, idx in enumerate(eval_indices):
        sample = ds[idx]
        with torch.no_grad():
            condition, generated_raw, generated, target_cpu = run_one_outpaint_eval(
                model,
                sample,
                text_pipe,
                perspective_size=a.perspective_size,
                num_inference_steps=a.num_inference_steps,
                inference_seed=a.eval_seed + i,
                inference_dtype=inference_dtype,
                eval_feather_sigma=a.eval_feather_sigma,
                eval_feather_kernel=a.eval_feather_kernel,
                inference_valid_mask_blur_kernel_px=a.eval_valid_mask_blur_kernel_px,
            )

        sid = ds.ids[idx]
        in_name = f"{i:03d}_{sid}_input.png"
        raw_name = f"{i:03d}_{sid}_generated_raw.png"
        gen_name = f"{i:03d}_{sid}_generated.png"
        tgt_name = f"{i:03d}_{sid}_target.png"
        out = a.output_dir
        if a.debug_masks:
            tp = sample.target_pixel_values.to(device=device, dtype=inference_dtype)
            vp = sample.view_params.to(device=device)
            _, unk_dbg = build_condition_from_target(
                target_pixel_values=tp,
                view_params=vp,
                num_views=sample.num_views,
                perspective_size=a.perspective_size,
            )
            u8 = (unk_dbg.detach().float().clamp(0, 1) * 255.0).byte().cpu().squeeze(0).numpy()
            mask_name = f"{i:03d}_{sid}_unknown_inpaint.png"
            Image.fromarray(u8, mode="L").save(os.path.join(out, mask_name))
            frac = float(unk_dbg.mean().item())
            print(f"[smoke][mask] {sid} mean(unknown)={frac:.4f} (1=inpaint; expect in (0,1))")
        Image.fromarray(_norm_to_uint8_img(condition.cpu())).save(os.path.join(out, in_name))
        Image.fromarray(_norm_to_uint8_img(generated_raw.cpu())).save(os.path.join(out, raw_name))
        Image.fromarray(_norm_to_uint8_img(generated.cpu())).save(os.path.join(out, gen_name))
        Image.fromarray(_norm_to_uint8_img(target_cpu)).save(os.path.join(out, tgt_name))
        rows.append(
            {
                "sample_id": sid,
                "input": in_name,
                "generated_raw": raw_name,
                "generated": gen_name,
                "target": tgt_name,
                "text": sample.captions,
            }
        )
        print(f"[smoke] OK sample {i}: {sid}")

    random.setstate(rng_state)
    html = _save_eval_html(a.output_dir, rows)
    print(f"[smoke] Done. Open: {html}")


if __name__ == "__main__":
    main()
