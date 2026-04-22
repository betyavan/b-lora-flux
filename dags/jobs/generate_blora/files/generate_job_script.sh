#!/bin/bash
set -e

# Env vars injected by Airflow:
#   EXPERIMENT_NAME           — e.g. e01_blora_flux_van_gogh_img1
#   TRAIN_OUTPUT_S3_PATH      — S3 path where LoRA weights were uploaded
#   GENERATED_OUTPUT_S3_PATH  — S3 path to upload generated images

# 0. Verify Python environment
cd /root/b-lora-flux
python scripts/check_env.py --strict

# 1. Symlink datasets
cp -r /my_datasets/data /root/b-lora-flux/data

cd /root/b-lora-flux

# 2. Download LoRA weights from S3
mkdir -p input_loras
s3cmd sync -v "${TRAIN_OUTPUT_S3_PATH%/}/" input_loras/

# 3. Find .safetensors
LORA_PATH=$(find input_loras -name "*.safetensors" | head -1)
if [ -z "$LORA_PATH" ]; then
  echo "ERROR: No .safetensors found in input_loras/"
  exit 1
fi
echo "Using LoRA: $LORA_PATH"

# 4. Generate 100 images
mkdir -p results/generated
python scripts/eval/generate_images.py \
  generate.lora_path="$LORA_PATH" \
  generate.prompt_file=data/coco_prompts.txt \
  generate.output_dir=results/generated \
  generate.exp_name="$EXPERIMENT_NAME"

# 5. Upload generated images to S3
s3cmd sync -v --follow-symlinks "results/generated/${EXPERIMENT_NAME}/" "${GENERATED_OUTPUT_S3_PATH%/}/"
