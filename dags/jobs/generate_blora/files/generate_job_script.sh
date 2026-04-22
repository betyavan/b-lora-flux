#!/bin/bash
set -e

# Populate s3cmd config with credentials from vault env vars
envsubst < /root/.s3cfg.template > /root/.s3cfg

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

# 2. Find .safetensors
LORA_PATH=$(find input_loras -name "*.safetensors" | head -1)
if [ -z "$LORA_PATH" ]; then
  echo "ERROR: No .safetensors found in input_loras/"
  exit 1
fi
echo "Using LoRA: $LORA_PATH"

# 3. Generate 100 images
mkdir -p results/generated
python scripts/eval/generate_images.py \
  generate.lora_path="$LORA_PATH" \
  generate.prompt_file=/my_datasets/coco_prompts.txt \
  generate.output_dir=results/generated \
  generate.exp_name="$EXPERIMENT_NAME"

# 4. Upload generated images to S3
s3cmd sync -v --follow-symlinks "results/generated/${EXPERIMENT_NAME}/" "${GENERATED_OUTPUT_S3_PATH%/}/"
