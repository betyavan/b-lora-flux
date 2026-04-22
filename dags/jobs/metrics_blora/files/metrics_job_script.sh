#!/bin/bash
set -e

# Populate s3cmd config with credentials from vault env vars
envsubst < /root/.s3cfg.template > /root/.s3cfg

# Env vars injected by Airflow:
#   EXPERIMENT_NAME           — e.g. e01_blora_flux_van_gogh_img1
#   GENERATED_OUTPUT_S3_PATH  — S3 path where generated images are stored
#   METRICS_OUTPUT_S3_PATH    — S3 path to upload metrics.json

# 0. Verify Python environment
cd /root/b-lora-flux
python scripts/check_env.py --strict

cd /root/b-lora-flux

# 1. Download generated images from S3
mkdir -p "results/generated/${EXPERIMENT_NAME}"
s3cmd sync -v "${GENERATED_OUTPUT_S3_PATH%/}/" "results/generated/${EXPERIMENT_NAME}/"

# 2. Determine style refs dir from experiment name
if echo "$EXPERIMENT_NAME" | grep -q "van_gogh"; then
  STYLE_REFS_DIR="/my_datasets/styles/van_gogh"
elif echo "$EXPERIMENT_NAME" | grep -q "monet"; then
  STYLE_REFS_DIR="/my_datasets/styles/monet"
else
  echo "WARNING: Cannot determine style from '${EXPERIMENT_NAME}', skipping style metrics"
  STYLE_REFS_DIR="null"
fi

# 3. Compute metrics
python scripts/eval/compute_metrics.py \
  metrics.generated_dir="results/generated/${EXPERIMENT_NAME}" \
  metrics.style_refs_dir="$STYLE_REFS_DIR" \
  metrics.prompt_file=/my_datasets/coco_prompts.txt \
  metrics.artbench_dir=/my_datasets/artbench10 \
  metrics.exp_name="$EXPERIMENT_NAME"

# 4. Upload metrics.json to S3
METRICS_JSON="output/results/${EXPERIMENT_NAME}.json"
if [ ! -f "$METRICS_JSON" ]; then
  echo "ERROR: metrics JSON not found at $METRICS_JSON"
  exit 1
fi
s3cmd put -v "$METRICS_JSON" "${METRICS_OUTPUT_S3_PATH%/}/metrics.json"
