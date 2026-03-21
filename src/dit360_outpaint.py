import copy

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
from src.outpaint_dataset import build_condition_from_target
from src.pipeline import DiT360Pipeline


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

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.flux_transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer"
        )
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

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
        self.weighting_scheme = args.weighting_scheme
        self.logit_mean = args.logit_mean
        self.logit_std = args.logit_std
        self.mode_scale = args.mode_scale

    def on_fit_start(self):
        self.vae = self.vae.to(dtype=torch.float32)

    def _forward_flux(self, noisy_model_input, timesteps, prompt_embeds, pooled_prompt_embeds, text_ids):
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

        latent_image_ids = DiT360Pipeline._prepare_latent_image_ids(
            bsz, packed_height, packed_width, self.device, self.dtype
        )
        if padding_n > 0:
            latent_image_ids = latent_image_ids.reshape(packed_height, packed_width, 3)
            first_col_image_ids = latent_image_ids[:, 0:padding_n, :]
            last_col_image_ids = latent_image_ids[:, -padding_n:, :]
            latent_image_ids = torch.cat([last_col_image_ids, latent_image_ids, first_col_image_ids], dim=1)
            latent_image_ids = latent_image_ids.reshape(-1, 3)

        if self.flux_transformer.config.guidance_embeds:
            guidance_vec = torch.full(
                (bsz,),
                self.hparams.args.guidance_scale,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            guidance_vec = None

        model_pred = self.flux_transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

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
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
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
    ) -> torch.Tensor:
        # tensors are expected on current model device
        bsz = condition_pixels.shape[0]
        condition_latents = encode_images(condition_pixels, self.vae).to(self.dtype)
        unknown_masks = F.interpolate(
            unknown_masks.to(self.dtype),
            size=(condition_latents.shape[2], condition_latents.shape[3]),
            mode="nearest",
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            condition_latents.shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        latents = unknown_masks * latents + (1.0 - unknown_masks) * condition_latents

        scheduler = copy.deepcopy(self.noise_scheduler)
        scheduler.set_timesteps(num_inference_steps, device=self.device)

        for t in scheduler.timesteps:
            timestep = t.expand(bsz).to(self.dtype)
            noisy_model_input = unknown_masks * latents + (1.0 - unknown_masks) * condition_latents
            model_pred = self._forward_flux(
                noisy_model_input=noisy_model_input,
                timesteps=timestep,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
            )
            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = unknown_masks * latents + (1.0 - unknown_masks) * condition_latents

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
        condition_latents = encode_images(condition_pixels, self.vae).to(self.dtype)
        unknown_masks = F.interpolate(
            unknown_masks,
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

        # Keep known pixels from condition latents, only denoise unknown region.
        noisy_model_input = unknown_masks * noisy_target + (1.0 - unknown_masks) * condition_latents

        model_pred = self._forward_flux(
            noisy_model_input=noisy_model_input,
            timesteps=timesteps,
            prompt_embeds=batch["prompt_embeds"],
            pooled_prompt_embeds=batch["pooled_prompt_embeds"],
            text_ids=batch["text_ids"],
        )

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        target = noise - target_latents

        per_pixel_loss = weighting.float() * (model_pred.float() - target.float()) ** 2
        masked_loss = per_pixel_loss * unknown_masks.float()
        denom = unknown_masks.sum().clamp(min=1.0)
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
