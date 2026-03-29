#!/bin/bash

set -euo pipefail

# Freeze IAM role credentials into env vars so DataLoader worker subprocesses
# (forked by PyTorch) can authenticate to S3 without hitting the metadata service.
eval $(python3 -c "
import boto3
c = boto3.Session().get_credentials().get_frozen_credentials()
print(f'export AWS_ACCESS_KEY_ID={c.access_key}')
print(f'export AWS_SECRET_ACCESS_KEY={c.secret_key}')
if c.token: print(f'export AWS_SESSION_TOKEN={c.token}')
")

# Base model: set USE_FILL_MODEL=1 to use FLUX Fill (inpaint) with 384-channel concat input.
export USE_FILL_MODEL=1
export FLUX="black-forest-labs/FLUX.1-Fill-dev"
# export FLUX="black-forest-labs/FLUX.1-dev"
export INIT_LORA="Insta360-Research/DiT360-Panorama-Image-Generation"

# Example:
# export TRAIN_IDS="./data/panomtdu_train.json"
# export PANO_ROOT="s3://your-bucket/your-pano-folder"
export TRAIN_IDS="./data/panomtdu_train.json"
export PANO_ROOT="s3://adobe-lingzhi-p/jingdongz-data/PanoPseudoLabels_processed/img"
export SAVE_DIR="outpaint_model_saved"
export TEST_IDS="/mnt/localssd/code/DiT360/data/panomtdu_test.json"
# export EVAL_EVERY=1000
# export EVAL_SAMPLES=8
# export EVAL_OUT_DIR="outpaint_eval_web"
# Inpaint mask dilation in full pano pixels before latent resize (0 = off). Default in Python is 4 if unset.
# export OUTPAINT_MASK_DILATE_PX=4
# RGB paste-back feather after eval sampling (pixels sigma); default 32 matches train_outpaint_lora.py.
# export EVAL_FEATHER_SIGMA=32
# Blur valid mask in pixel space before latent sampling (sample_outpaint only); even sizes use k+1.
# export EVAL_VALID_MASK_BLUR_KERNEL_PX=32
# Optional subset mix ratio example:
export SUBSET_RATIOS="Sun360:2,ZInD:1,scene:2,Matterport3D:4,Hunyuan:1"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_DEVICE=8

python train_outpaint_lora.py \
  --pretrained_model_name_or_path="$FLUX" \
  --init_lora_weights="$INIT_LORA" \
  --train_id_list="$TRAIN_IDS" \
  --test_id_list="$TEST_IDS" \
  --pano_root="$PANO_ROOT" \
  --image_ext="jpg" \
  --exclude_prefixes "DiT360_" \
  --seed=0 \
  --pano_width=1024 \
  --pano_height=512 \
  --perspective_size=512 \
  --min_views=1 \
  --max_views=10 \
  --min_fov=75 \
  --max_fov=105 \
  --min_pitch=-75 \
  --max_pitch=75 \
  --train_batch_size=1 \
  --adam_weight_decay=1e-2 \
  --dataloader_num_workers=16 \
  --subset_sampling_ratios="${SUBSET_RATIOS:-Sun360:4,ZInD:1,scene:2,Matterport3D:4,Hunyuan:1}" \
  --save_dir="$SAVE_DIR" \
  --devices="$CUDA_DEVICE" \
  --max_steps="${MAX_STEPS:-10000}" \
  --warmup_steps="${WARMUP_STEPS:-500}" \
  --lr_scheduler="${LR_SCHED:-cosine}" \
  --guidance_scale=1.0 \
  --rank=64 \
  --lora_alpha=64 \
  --lora_drop_out=0.05 \
  --gaussian_init_lora \
  --precision=bf16-mixed \
  --accumulate_grad_batches=1 \
  --learning_rate="${LR:-2e-5}" \
  --adam_epsilon=1e-6 \
  --padding_n=1 \
  --save_every_n_steps="${SAVE_STEPS:-1000}" \
  --eval_every_n_steps="${EVAL_EVERY:-500}" \
  --eval_num_samples="${EVAL_SAMPLES:-20}" \
  --eval_output_dir="${EVAL_OUT_DIR:-outpaint_eval_web}" \
  --eval_inference_steps=30 \
  --eval_feather_sigma="${EVAL_FEATHER_SIGMA:-32}" \
  --eval_valid_mask_blur_kernel_px="${EVAL_VALID_MASK_BLUR_KERNEL_PX:-32}" \
  --outpaint_mask_dilate_px="${OUTPAINT_MASK_DILATE_PX:-4}" \
  ${USE_FILL_MODEL:+--use_fill_model}
