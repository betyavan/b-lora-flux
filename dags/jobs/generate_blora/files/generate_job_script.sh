#!/bin/bash
set -e

# Env vars injected by Airflow:
#   EXPERIMENT_NAME           — e.g. e01_blora_flux_van_gogh_img1
#   TRAIN_OUTPUT_S3_PATH      — S3 path where LoRA weights were uploaded
#   GENERATED_OUTPUT_S3_PATH  — S3 path to upload generated images

# 0. Verify Python environment
cd /root/b-lora-flux
python scripts/check_env.py --strict

cd /root/b-lora-flux

# 1. Download LoRA weights from S3
mkdir -p input_loras
s3cmd sync -v "${TRAIN_OUTPUT_S3_PATH%/}/" input_loras/

# 2. Find .safetensors (optional — baseline experiments have no LoRA)
LORA_PATH=$(find input_loras -name "*.safetensors" | head -1)
if [ -z "$LORA_PATH" ]; then
  echo "WARNING: No .safetensors found — running as baseline (no LoRA)"
  LORA_ARG="generate.lora_path=null"
else
  echo "Using LoRA: $LORA_PATH"
  LORA_ARG="generate.lora_path=${LORA_PATH}"
fi

# 3. Generate 100 images
mkdir -p results/generated
python scripts/eval/generate_images.py \
  "$LORA_ARG" \
  generate.prompt_file=/my_datasets/coco_prompts.txt \
  generate.output_dir=results/generated \
  generate.exp_name="$EXPERIMENT_NAME" \
  model.lora_scale="${LORA_SCALE:-1.0}"

# 4. Upload generated images to S3
s3cmd sync -v --follow-symlinks "results/generated/${EXPERIMENT_NAME}/" "${GENERATED_OUTPUT_S3_PATH%/}/"
