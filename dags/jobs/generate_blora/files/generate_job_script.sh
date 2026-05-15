#!/bin/bash
set -e

# Env vars injected by Airflow:
#   EXPERIMENT_NAME           — e.g. e01_blora_flux_van_gogh_img1
#   TRAIN_OUTPUT_S3_PATH      — S3 path where LoRA weights were uploaded
#   GENERATED_OUTPUT_S3_PATH  — S3 path to upload generated images
#   PROMPT_SUFFIX             — (optional) appended to every COCO prompt, e.g. "painted in the style of Van Gogh"

# 0. Verify Python environment
cd /root/b-lora-flux
python scripts/check_env.py --strict

# Pull eval data via DVC
dvc pull data/coco_prompts.txt.dvc

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

# Detect SDXL from experiment config (is_xl: true → use SDXL pipeline and model path)
IS_XL=$(python3 -c "
import yaml, sys
try:
    with open('configs/experiments/${EXPERIMENT_NAME}.yaml') as f:
        cfg = yaml.safe_load(f)
    proc = (cfg.get('config', {}).get('process') or [{}])[0]
    print('true' if proc.get('model', {}).get('is_xl') else 'false')
except Exception:
    print('false')
")

if [ "$IS_XL" = "true" ]; then
  echo "INFO: SDXL experiment detected — using SDXL pipeline"
  MODEL_ARGS="model.name_or_path=/models/sdxl model.pipeline_type=sdxl sampling.steps=20 sampling.guidance_scale=7.5"
else
  MODEL_ARGS=""
fi

# 3. Generate 100 images
mkdir -p results/generated
python scripts/eval/generate_images.py \
  "$LORA_ARG" \
  generate.prompt_file=data/coco_prompts.txt \
  "generate.prompt_suffix=${PROMPT_SUFFIX:-}" \
  generate.output_dir=results/generated \
  generate.exp_name="$EXPERIMENT_NAME" \
  model.lora_scale="${LORA_SCALE:-1.0}" \
  ${MODEL_ARGS}

# 4. Upload generated images to S3
s3cmd sync -v --follow-symlinks "results/generated/${EXPERIMENT_NAME}/" "${GENERATED_OUTPUT_S3_PATH%/}/"
