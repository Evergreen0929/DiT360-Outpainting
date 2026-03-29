import argparse
import os
import random
from functools import partial
from typing import Dict, List

import lightning as L
import torch
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from peft import LoraConfig
from pytorch_lightning import seed_everything
from torch.utils.data import WeightedRandomSampler

from src.dit360_outpaint import DiT360Outpaint
from src.outpaint_dataset import RandomPerspOutpaintDataset
from src.outpaint_eval_utils import run_one_outpaint_eval
from src.pipeline import DiT360Pipeline

class VerboseTQDMProgressBar(TQDMProgressBar):

    def __init__(self, decimals: int = 8, **kwargs):
        super().__init__(**kwargs)
        self._decimals = decimals

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        fmt = f"{{:.{self._decimals}f}}"
        out = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.detach().item()
            if isinstance(v, float):
                out[k] = fmt.format(v)
            else:
                out[k] = v
        return out

def collate_fn(examples, text_encoding_pipeline):
    examples = list(examples)
    pixel_values = torch.stack([cast.target_pixel_values for cast in examples])
    view_params = torch.stack([cast.view_params for cast in examples])
    num_views = torch.tensor([cast.num_views for cast in examples], dtype=torch.long)
    captions = [cast.captions for cast in examples]

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(captions, prompt_2=None)

    return {
        "target_pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "view_params": view_params.float(),
        "num_views": num_views,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }


def _norm_to_uint8_img(tensor: torch.Tensor):
    x = tensor.detach().cpu().float().clamp(-1.0, 1.0) 
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")


def _save_eval_html(output_dir: str, step: int, rows: List[Dict[str, str]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"step_{step:08d}.html")
    parts = [
        "<html><head><meta charset='utf-8'><title>Outpaint Eval</title>",
        "<style>body{font-family:Arial,sans-serif;padding:20px;} table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ccc;padding:8px;vertical-align:top;} img{max-width:100%;height:auto;}</style>",
        "</head><body>",
        f"<h2>Outpaint Eval - step {step}</h2>",
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


class PeriodicOutpaintEvalCallback(Callback):
    def __init__(self, args, text_encoding_pipeline):
        super().__init__()
        self.args = args
        self.text_encoding_pipeline = text_encoding_pipeline
        self.eval_dataset = RandomPerspOutpaintDataset(
            id_list_file=args.test_id_list,
            pano_root=args.pano_root,
            pano_height=args.pano_height,
            pano_width=args.pano_width,
            image_ext=args.image_ext,
            caption_map_file=args.caption_map_file,
            caption_template=args.caption_template,
            exclude_prefixes=args.exclude_prefixes,
            min_views=args.min_views,
            max_views=args.max_views,
            min_fov=args.min_fov,
            max_fov=args.max_fov,
            min_pitch=args.min_pitch,
            max_pitch=args.max_pitch,
            perspective_size=args.perspective_size,
        )

        total = len(self.eval_dataset)
        k = min(args.eval_num_samples, total)
        rng = random.Random(args.eval_seed)
        self.eval_indices = rng.sample(range(total), k=k)
        print(f"[Eval] Subset: {k}/{total} dataset indices (random.sample, eval_seed={args.eval_seed})")

    @torch.no_grad()
    def _run_eval(self, trainer, pl_module, step: int):
        pl_module.eval()
        device = pl_module.device
        step_dir = os.path.join(self.args.eval_output_dir, f"step_{step:08d}")
        os.makedirs(step_dir, exist_ok=True)

        eval_rng_state = random.getstate()
        random.seed(self.args.eval_seed)

        inference_dtype = pl_module.dtype
        rows = []
        for i, idx in enumerate(self.eval_indices):
            sample = self.eval_dataset[idx]
            condition, generated_raw, _generated_composited, target_cpu = run_one_outpaint_eval(
                pl_module,
                sample,
                self.text_encoding_pipeline,
                perspective_size=self.args.perspective_size,
                num_inference_steps=self.args.eval_inference_steps,
                inference_seed=self.args.eval_seed + i,
                inference_dtype=inference_dtype,
                eval_feather_sigma=self.args.eval_feather_sigma,
                eval_feather_kernel=self.args.eval_feather_kernel,
                inference_valid_mask_blur_kernel_px=self.args.eval_valid_mask_blur_kernel_px,
            )

            sid = self.eval_dataset.ids[idx]
            in_name = f"{i:03d}_{sid}_input.png"
            gen_name = f"{i:03d}_{sid}_generated.png"
            tgt_name = f"{i:03d}_{sid}_target.png"

            Image.fromarray(_norm_to_uint8_img(condition.cpu())).save(os.path.join(step_dir, in_name))
            Image.fromarray(_norm_to_uint8_img(generated_raw.cpu())).save(os.path.join(step_dir, gen_name))
            Image.fromarray(_norm_to_uint8_img(target_cpu)).save(os.path.join(step_dir, tgt_name))

            rows.append(
                {
                    "sample_id": sid,
                    "input": in_name,
                    "text": sample.captions,
                    "generated": gen_name,
                    "target": tgt_name,
                }
            )

        random.setstate(eval_rng_state)

        html_path = _save_eval_html(step_dir, step, rows)
        print(f"[Eval] Saved periodic eval webpage to: {html_path}")
        pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.args.eval_every_n_steps <= 0:
            return
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.args.eval_every_n_steps != 0:
            return
        self._run_eval(trainer, pl_module, step)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--use_fill_model",
        action="store_true",
        help="Use FLUX Fill (inpaint) model with 384-channel concat input (noisy_latent + masked_image_latent + mask). "
        "Base model should be black-forest-labs/FLUX.1-Fill-dev.",
    )
    parser.add_argument("--init_lora_weights", type=str, default="Insta360-Research/DiT360-Panorama-Image-Generation")
    parser.add_argument("--train_id_list", type=str, required=True, help="Path to json list with pano ids.")
    parser.add_argument("--caption_map_file", type=str, default=None, help="Optional JSON map: pano_id -> caption.")
    parser.add_argument("--pano_root", type=str, required=True, help="Root folder, supports s3://bucket/path.")
    parser.add_argument("--image_ext", type=str, default="jpg")
    parser.add_argument("--caption_template", type=str, default="This is a panorama image.")
    parser.add_argument(
        "--exclude_prefixes",
        type=str,
        nargs="+",
        default=["DiT360_"],
        help="Any sample id starting with one of these prefixes will be ignored.",
    )

    parser.add_argument("--pano_height", type=int, default=512)
    parser.add_argument("--pano_width", type=int, default=1024)
    parser.add_argument("--perspective_size", type=int, default=512)
    parser.add_argument(
        "--outpaint_mask_dilate_px",
        type=int,
        default=4,
        help="Dilate unknown mask in pixel space (hard edges) before nearest resize to latent, train + sample_outpaint; 0=off. "
        "Eval RGB composite still uses the undilated mask.",
    )
    parser.add_argument("--min_views", type=int, default=1)
    parser.add_argument("--max_views", type=int, default=10)
    parser.add_argument("--min_fov", type=float, default=75.0)
    parser.add_argument("--max_fov", type=float, default=105.0)
    parser.add_argument("--min_pitch", type=float, default=-30.0)
    parser.add_argument("--max_pitch", type=float, default=30.0)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--guidance_scale", type=float, default=1.0)

    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--gaussian_init_lora", action="store_true")
    parser.add_argument("--lora_drop_out", type=float, default=0.05)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument(
        "--subset_sampling_ratios",
        type=str,
        default="Sun360:2,ZInD:1,scene:2,Matterport3D:4,Hunyuan:1",
        help="Optional subset ratio spec, e.g. 'Sun360:1,ZInD:2,scene:1'.",
    )
    parser.add_argument("--save_dir", type=str, default="outpaint_model_saved")
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Max epochs; ignored when --max_steps is set (>0).")
    parser.add_argument("--max_steps", type=int, default=30000, help="Total optimizer steps. -1 means use max_epochs.")
    parser.add_argument("--warmup_steps", type=int, default=-1, help="Warmup steps. Default -1 = 5%% of max_steps.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "constant", "constant_with_warmup"])
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--save_every_n_steps", type=int, default=5000, help="Save checkpoint every N optimizer steps.")
    parser.add_argument("--padding_n", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--test_id_list", type=str, default="/mnt/localssd/code/DiT360/data/panomtdu_test.json")
    parser.add_argument("--eval_every_n_steps", type=int, default=1000)
    parser.add_argument("--eval_num_samples", type=int, default=20)
    parser.add_argument("--eval_output_dir", type=str, default="outpaint_eval_web")
    parser.add_argument("--eval_inference_steps", type=int, default=30)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=1234,
        help="RNG seed for eval: which test IDs are chosen (random.sample) and view randomness during eval.",
    )
    parser.add_argument(
        "--eval_feather_sigma",
        type=float,
        default=32.0,
        help="Eval only: Gaussian sigma (pixels) when pasting known region from condition onto generated; 0 = hard mask.",
    )
    parser.add_argument(
        "--eval_feather_kernel",
        type=int,
        default=None,
        help="Eval only: odd blur kernel size for feather; default derived from sigma.",
    )
    parser.add_argument(
        "--eval_valid_mask_blur_kernel_px",
        type=int,
        default=32,
        help="Eval/sample_outpaint only: Gaussian blur valid (known) mask in pixel space before latent resize; "
        "<3 disables. Training loss path unchanged.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    text_encoding_pipeline = DiT360Pipeline.from_pretrained(args.pretrained_model_name_or_path, vae=None, transformer=None)

    train_dataset = RandomPerspOutpaintDataset(
        id_list_file=args.train_id_list,
        pano_root=args.pano_root,
        pano_height=args.pano_height,
        pano_width=args.pano_width,
        image_ext=args.image_ext,
        caption_map_file=args.caption_map_file,
        caption_template=args.caption_template,
        exclude_prefixes=args.exclude_prefixes,
        min_views=args.min_views,
        max_views=args.max_views,
        min_fov=args.min_fov,
        max_fov=args.max_fov,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
        perspective_size=args.perspective_size,
    )

    sampler = None
    shuffle = True
    if args.subset_sampling_ratios:
        weights = train_dataset.build_sample_weights(args.subset_sampling_ratios)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)
        shuffle = False
        print(f"[SubsetRatio] Using subset sampling ratios: {args.subset_sampling_ratios}")
        print(f"[SubsetRatio] Subset counts: {train_dataset.subset_counts}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        drop_last=True,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=partial(collate_fn, text_encoding_pipeline=text_encoding_pipeline),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )

    target_modules = ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0"]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        bias=args.lora_bias,
        lora_dropout=args.lora_drop_out,
    )

    epoch_ckpt = ModelCheckpoint(
        save_top_k=args.topk,
        every_n_epochs=1,
        filename="outpaint_{epoch:02d}_{train_loss:.4f}",
        save_last=True,
    )
    step_ckpt = ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=args.save_every_n_steps,
        filename="outpaint_step_{step:08d}",
    )
    eval_callback = PeriodicOutpaintEvalCallback(args=args, text_encoding_pipeline=text_encoding_pipeline)

    csv_logger = CSVLogger(save_dir=args.save_dir, name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=args.save_dir, name="tb_logs")

    use_max_steps = args.max_steps > 0
    trainer = L.Trainer(
        devices=args.devices,
        accelerator="gpu",
        callbacks=[VerboseTQDMProgressBar(decimals=8), epoch_ckpt, step_ckpt, eval_callback],
        logger=[tb_logger, csv_logger],
        default_root_dir=args.save_dir,
        max_steps=args.max_steps if use_max_steps else -1,
        max_epochs=args.max_epochs if not use_max_steps else -1,
        precision=args.precision,
        strategy="deepspeed_stage_2",
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
    )

    model = DiT360Outpaint(args, lora_config=lora_config)
    trainer.fit(model, train_dataloader, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
