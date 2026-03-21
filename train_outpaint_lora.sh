#!/bin/bash

set -euo pipefail

export FLUX="black-forest-labs/FLUX.1-dev"
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
  --subset_sampling_ratios="${SUBSET_RATIOS:-Sun360:2,ZInD:1,scene:2,Matterport3D:4,Hunyuan:1}" \
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
  --eval_every_n_steps="${EVAL_EVERY:-1000}" \
  --eval_num_samples="${EVAL_SAMPLES:-20}" \
  --eval_output_dir="${EVAL_OUT_DIR:-outpaint_eval_web}" \
  --eval_inference_steps=30
