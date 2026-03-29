import copy
from contextlib import nullcontext
from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from src.lora_init import load_initial_lora_weights
from src.outpaint_dataset import build_condition_from_target, dilate_unknown_mask
from src.outpaint_eval_utils import _gaussian_blur2d_1ch
from src.pipeline import DiT360Pipeline, calculate_shift, retrieve_timesteps


def _soften_unknown_mask_by_blurring_valid(um: torch.Tensor, kernel_px: int) -> torch.Tensor:
    """
    Pixel-space: Gaussian-blur the valid (known) mask ``valid = 1 - unknown``, then return new
    unknown = 1 - blurred(valid). Softens the inpaint boundary for inference only; odd kernel if
    ``kernel_px`` is even.
    um: [B, 1, H, W] unknown in [0, 1].
    """
    if kernel_px < 3:
        return um
    k = kernel_px if kernel_px % 2 == 1 else kernel_px + 1
    sigma = float(k - 1) / 6.0
    valid = (1.0 - um).clamp(0.0, 1.0)
    blurred_valid = _gaussian_blur2d_1ch(valid, k, sigma).clamp(0.0, 1.0)
    return (1.0 - blurred_valid).clamp(0.0, 1.0)


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def get_sigmas(noise_scheduler_copy, timesteps, device, dtype, n_dim=4):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device=device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class DiT360Outpaint(L.LightningModule):
    def __init__(self, args, lora_config):
        super().__init__()
        self.save_hyperparameters()
        args = self.hparams.args

        self._use_fill_model = getattr(args, "use_fill_model", False)

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.flux_transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer"
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        if self._use_fill_model:
            expected_in_ch = 384
            actual_in_ch = self.flux_transformer.config.in_channels
            if actual_in_ch != expected_in_ch:
                raise ValueError(
                    f"--use_fill_model expects transformer in_channels={expected_in_ch} "
                    f"(FLUX Fill), but loaded model has in_channels={actual_in_ch}. "
                    f"Use black-forest-labs/FLUX.1-Fill-dev as pretrained_model_name_or_path."
                )
            print(f"[FillModel] Using FLUX Fill model with in_channels={actual_in_ch}")

        self.vae.requires_grad_(False)
        self.flux_transformer.requires_grad_(False)
        self.flux_transformer.add_adapter(lora_config)

        if args.init_lora_weights:
            load_initial_lora_weights(self.flux_transformer, args.init_lora_weights)

        for param in self.flux_transformer.parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype=torch.float32)

        self.flux_transformer.train()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self._mask_dilate_px = int(getattr(args, "outpaint_mask_dilate_px", 0) or 0)
        self.weighting_scheme = args.weighting_scheme
        self.logit_mean = args.logit_mean
        self.logit_std = args.logit_std
        self.mode_scale = args.mode_scale

    def on_fit_start(self):
        self.vae = self.vae.to(dtype=torch.float32)

    def _prepare_fill_condition(
        self,
        condition_pixels: torch.Tensor,
        unknown_masks_pixel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare FLUX Fill-style packed condition tensor.

        Matches ``FluxFillPipeline.prepare_mask_latents`` exactly:
        1. masked_image = condition * (1 - mask)  →  VAE encode  →  pack  →  [B, seq, 64]
        2. mask [B, 1, H_px, W_px] →  reshape to [B, vae_sf², H_lat, W_lat]  →  pack  →  [B, seq, 256]
        3. concat  →  [B, seq, 320]

        Args:
            condition_pixels: [B, 3, H_px, W_px] in [-1, 1], unknown regions may be any value.
            unknown_masks_pixel: [B, 1, H_px, W_px] in {0, 1}, 1 = inpaint/unknown.
        Returns:
            Packed fill condition [B, seq, 320] (not yet padded for wraparound).
        """
        # 1. Masked image: zero-out unknown regions (matches FLUX Fill: image * (1 - mask))
        masked_image = condition_pixels * (1.0 - unknown_masks_pixel)

        # 2. VAE encode masked image
        masked_image_latents = encode_images(masked_image, self.vae)  # [B, C_vae, H_lat, W_lat]
        bsz, c_vae, h_lat, w_lat = masked_image_latents.shape

        # 3. Pack masked_image_latents → [B, seq, c_vae*4]
        packed_mi = DiT360Pipeline._pack_latents(
            masked_image_latents, bsz, c_vae, h_lat, w_lat,
        )  # [B, seq, 64]

        # 4. Prepare mask in FLUX Fill format:
        #    pixel mask → [B, vae_sf², H_lat, W_lat] → pack → [B, seq, vae_sf²*4]
        sf = self.vae_scale_factor  # 8
        mask = unknown_masks_pixel[:, 0, :, :]  # [B, H_px, W_px]
        mask = mask.view(bsz, h_lat, sf, w_lat, sf)
        mask = mask.permute(0, 2, 4, 1, 3)  # [B, sf, sf, H_lat, W_lat]
        mask = mask.reshape(bsz, sf * sf, h_lat, w_lat)  # [B, 64, H_lat, W_lat]
        packed_mask = DiT360Pipeline._pack_latents(
            mask, bsz, sf * sf, h_lat, w_lat,
        )  # [B, seq, 256]

        # 5. Concat → [B, seq, 320]
        return torch.cat([packed_mi, packed_mask], dim=-1)

    def _forward_flux(
        self,
        noisy_model_input,
        timesteps,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        forward_dtype: Optional[torch.dtype] = None,
        guidance_scale_override: Optional[float] = None,
        fill_condition_packed: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            noisy_model_input: [B, C_vae, H_lat, W_lat] noisy latents (C_vae=16 for FLUX).
            fill_condition_packed: (Fill mode only) [B, seq, 320] from ``_prepare_fill_condition``,
                already packed but **not** padded for wraparound. Will be padded and concatenated
                with the packed noisy latents to form [B, padded_seq, 384].
        """
        dt = forward_dtype if forward_dtype is not None else self.dtype
        bsz = noisy_model_input.shape[0]
        packed_noisy_model_input = DiT360Pipeline._pack_latents(
            noisy_model_input,
            batch_size=bsz,
            num_channels_latents=noisy_model_input.shape[1],
            height=noisy_model_input.shape[2],
            width=noisy_model_input.shape[3],
        )

        padding_n = self.hparams.args.padding_n
        packed_height = noisy_model_input.shape[2] // 2
        packed_width = noisy_model_input.shape[3] // 2
        dim = packed_noisy_model_input.shape[-1]

        if padding_n > 0:
            packed_noisy_model_input = packed_noisy_model_input.reshape(bsz, packed_height, packed_width, dim)
            first_col = packed_noisy_model_input[:, :, 0:padding_n, :]
            last_col = packed_noisy_model_input[:, :, -padding_n:, :]
            packed_noisy_model_input = torch.cat([last_col, packed_noisy_model_input, first_col], dim=2)
            packed_noisy_model_input = packed_noisy_model_input.reshape(bsz, -1, dim)

            # Apply identical wraparound padding to fill_condition_packed
            if fill_condition_packed is not None:
                fill_dim = fill_condition_packed.shape[-1]
                fill_condition_packed = fill_condition_packed.reshape(bsz, packed_height, packed_width, fill_dim)
                fc_first = fill_condition_packed[:, :, 0:padding_n, :]
                fc_last = fill_condition_packed[:, :, -padding_n:, :]
                fill_condition_packed = torch.cat([fc_last, fill_condition_packed, fc_first], dim=2)
                fill_condition_packed = fill_condition_packed.reshape(bsz, -1, fill_dim)

        # For Fill mode: concatenate [noisy_latents, fill_condition] along feature dim → [B, seq, 384]
        if fill_condition_packed is not None:
            packed_noisy_model_input = torch.cat(
                [packed_noisy_model_input, fill_condition_packed], dim=2,
            )

        latent_image_ids = DiT360Pipeline._prepare_latent_image_ids(
            bsz, packed_height, packed_width, self.device, dt
        )
        if padding_n > 0:
            latent_image_ids = latent_image_ids.reshape(packed_height, packed_width, 3)
            first_col_image_ids = latent_image_ids[:, 0:padding_n, :]
            last_col_image_ids = latent_image_ids[:, -padding_n:, :]
            latent_image_ids = torch.cat([last_col_image_ids, latent_image_ids, first_col_image_ids], dim=1)
            latent_image_ids = latent_image_ids.reshape(-1, 3)

        if self.flux_transformer.config.guidance_embeds:
            g = (
                guidance_scale_override
                if guidance_scale_override is not None
                else self.hparams.args.guidance_scale
            )
            guidance_vec = torch.full(
                (bsz,),
                g,
                device=self.device,
                dtype=dt,
            )
        else:
            guidance_vec = None

        target_dtype = torch.bfloat16 if "bf16" in str(self.hparams.args.precision) else torch.float16
        t_norm = timesteps / 1000

        hs_in = packed_noisy_model_input.to(target_dtype)
        t_in = t_norm.to(target_dtype)
        img_in = latent_image_ids.to(target_dtype)
        gv_in = guidance_vec.to(target_dtype) if guidance_vec is not None else None
        pe_in = prompt_embeds.to(target_dtype)
        ppe_in = pooled_prompt_embeds.to(target_dtype)
        tid_in = text_ids.to(target_dtype)

        amp_ctx = torch.autocast(device_type="cuda", dtype=target_dtype) if self.device.type == "cuda" else nullcontext()

        with amp_ctx:
            model_pred = self.flux_transformer(
                hidden_states=hs_in,
                timestep=t_in,
                guidance=gv_in,
                pooled_projections=ppe_in,
                encoder_hidden_states=pe_in,
                txt_ids=tid_in,
                img_ids=img_in,
                return_dict=False,
            )[0]

        if model_pred.dtype != dt:
            model_pred = model_pred.to(dt)

        if padding_n > 0:
            model_pred = model_pred.reshape(bsz, packed_height, packed_width + 2 * padding_n, -1)
            model_pred = model_pred[:, :, padding_n:-padding_n, :]
            model_pred = model_pred.reshape(bsz, packed_height * packed_width, -1)

        model_pred = DiT360Pipeline._unpack_latents(
            model_pred,
            height=noisy_model_input.shape[2] * self.vae_scale_factor,
            width=noisy_model_input.shape[3] * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        return model_pred

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # VAE is kept in float32 (on_fit_start); latents from sampling may be fp16 under mixed precision.
        vae_dtype = next(self.vae.parameters()).dtype
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        latents = latents.to(dtype=vae_dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        return image.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_outpaint(
        self,
        condition_pixels: torch.Tensor,
        unknown_masks: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        num_inference_steps: int = 16,
        seed: int = 0,
        inference_dtype: Optional[torch.dtype] = None,
        guidance_scale_override: Optional[float] = None,
        valid_mask_blur_kernel_px: int = 0,
    ) -> torch.Tensor:
        # tensors are expected on current model device
        dt = inference_dtype if inference_dtype is not None else self.dtype
        bsz = condition_pixels.shape[0]
        amp_ctx = torch.autocast(device_type="cuda", dtype=dt) if self.device.type == "cuda" else nullcontext()

        um = unknown_masks.to(dt)
        if self._mask_dilate_px > 0:
            um = dilate_unknown_mask(um, radius_px=self._mask_dilate_px)

        # Fill mode: prepare condition with the hard (dilated-only) mask before any blur.
        # _soften_unknown_mask_by_blurring_valid was designed for standard mode, where a soft mask
        # smooths the noisy/clean latent compositing boundary. In Fill mode it would partially zero
        # clean known pixels (masked_image = condition * (1 - soft_mask)), which misaligns with how
        # FLUX Fill was trained (hard binary masks). Dilation is still applied above.
        fill_condition_packed = None
        if self._use_fill_model:
            fill_condition_packed = self._prepare_fill_condition(
                condition_pixels, um,
            ).to(dt)

        # Standard mode: optionally blur the mask for soft latent compositing boundary.
        if not self._use_fill_model and valid_mask_blur_kernel_px >= 3:
            um = _soften_unknown_mask_by_blurring_valid(um, valid_mask_blur_kernel_px)

        condition_latents = encode_images(condition_pixels, self.vae).to(dt)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            condition_latents.shape,
            device=self.device,
            dtype=dt,
            generator=generator,
        )
        if not self._use_fill_model:
            unknown_masks_latent = F.interpolate(
                um,
                size=(condition_latents.shape[2], condition_latents.shape[3]),
                mode="nearest",
            )
            latents = unknown_masks_latent * latents + (1.0 - unknown_masks_latent) * condition_latents

        scheduler = copy.deepcopy(self.noise_scheduler)
        _, _, h_lat, w_lat = condition_latents.shape
        packed = DiT360Pipeline._pack_latents(
            condition_latents,
            batch_size=bsz,
            num_channels_latents=condition_latents.shape[1],
            height=h_lat,
            width=w_lat,
        )
        image_seq_len = packed.shape[1]
        sched_cfg = scheduler.config
        timestep_kwargs = {}
        if getattr(sched_cfg, "use_dynamic_shifting", False):
            timestep_kwargs["mu"] = calculate_shift(
                image_seq_len,
                getattr(sched_cfg, "base_image_seq_len", 256),
                getattr(sched_cfg, "max_image_seq_len", 4096),
                getattr(sched_cfg, "base_shift", 0.5),
                getattr(sched_cfg, "max_shift", 1.15),
            )
        timesteps, _ = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            self.device,
            **timestep_kwargs,
        )

        with amp_ctx:
            for t in timesteps:
                timestep = t.expand(bsz).to(dt)
                if self._use_fill_model:
                    # Fill mode: pass full noisy latents; condition via concatenated channels.
                    noisy_model_input = latents
                else:
                    # Standard mode: re-composite known region at each step.
                    noisy_model_input = unknown_masks_latent * latents + (1.0 - unknown_masks_latent) * condition_latents  # noqa: F821
                model_pred = self._forward_flux(
                    noisy_model_input=noisy_model_input,
                    timesteps=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    forward_dtype=dt,
                    guidance_scale_override=guidance_scale_override,
                    fill_condition_packed=fill_condition_packed,
                )
                latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
                if not self._use_fill_model:
                    latents = unknown_masks_latent * latents + (1.0 - unknown_masks_latent) * condition_latents  # noqa: F821

        return self._decode_latents(latents)

    def training_step(self, batch, batch_idx):
        target_pixels = batch["target_pixel_values"].to(device=self.device, dtype=self.dtype)
        view_params = batch["view_params"].to(device=self.device, dtype=torch.float32)
        num_views = batch["num_views"].to(device=self.device)

        with torch.no_grad():
            condition_list = []
            unknown_list = []
            for i in range(target_pixels.shape[0]):
                cond_i, unk_i = build_condition_from_target(
                    target_pixel_values=target_pixels[i],
                    view_params=view_params[i],
                    num_views=int(num_views[i].item()),
                    perspective_size=self.hparams.args.perspective_size,
                )
                condition_list.append(cond_i)
                unknown_list.append(unk_i)

            condition_pixels = torch.stack(condition_list, dim=0).to(device=self.device, dtype=self.dtype)
            unknown_masks = torch.stack(unknown_list, dim=0).to(device=self.device, dtype=self.dtype)

        target_latents = encode_images(target_pixels, self.vae).to(self.dtype)

        # Prepare mask (pixel-space dilation, then latent-space resize)
        um = unknown_masks
        if self._mask_dilate_px > 0:
            um = dilate_unknown_mask(um, radius_px=self._mask_dilate_px)

        # For Fill mode: prepare condition through FLUX Fill concatenation path
        fill_condition_packed = None
        if self._use_fill_model:
            fill_condition_packed = self._prepare_fill_condition(
                condition_pixels, um,
            ).to(self.dtype)

        # Standard mode still needs condition_latents and latent-space unknown mask
        if not self._use_fill_model:
            condition_latents = encode_images(condition_pixels, self.vae).to(self.dtype)

        unknown_masks_latent = F.interpolate(
            um,
            size=(target_latents.shape[2], target_latents.shape[3]),
            mode="nearest",
        )

        bsz = target_latents.shape[0]
        noise = torch.randn_like(target_latents, device=self.device, dtype=self.dtype)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )

        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=self.device)

        sigmas = get_sigmas(
            self.noise_scheduler_copy,
            timesteps,
            device=self.device,
            dtype=self.dtype,
            n_dim=target_latents.ndim,
        )
        noisy_target = (1.0 - sigmas) * target_latents + sigmas * noise

        if self._use_fill_model:
            # Fill mode: full noisy target as input; condition through concatenated channels.
            noisy_model_input = noisy_target
        else:
            # Standard mode: keep known pixels from condition latents, only denoise unknown region.
            noisy_model_input = unknown_masks_latent * noisy_target + (1.0 - unknown_masks_latent) * condition_latents

        model_pred = self._forward_flux(
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            prompt_embeds=batch["prompt_embeds"],
            pooled_prompt_embeds=batch["pooled_prompt_embeds"],
            text_ids=batch["text_ids"],
            fill_condition_packed=fill_condition_packed,
        )

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        target = noise - target_latents

        per_pixel_loss = weighting.float() * (model_pred.float() - target.float()) ** 2
        masked_loss = per_pixel_loss * unknown_masks_latent.float()
        denom = unknown_masks_latent.float().sum().clamp(min=1.0)
        loss = masked_loss.sum() / (denom * target.shape[1])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        args = self.hparams.args
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.flux_transformer.parameters()))
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

        max_steps = args.max_steps if args.max_steps > 0 else 30000
        warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.05 * max_steps)

        if args.lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        elif args.lr_scheduler == "constant_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
