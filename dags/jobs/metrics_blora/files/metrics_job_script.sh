#!/bin/bash
set -e

# Env vars injected by Airflow:
#   EXPERIMENT_NAME           — e.g. e01_blora_flux_van_gogh_img1
#   GENERATED_OUTPUT_S3_PATH  — S3 path where generated images are stored
#   METRICS_OUTPUT_S3_PATH    — S3 path to upload metrics.json

# 0. Verify Python environment
cd /root/b-lora-flux
python scripts/check_env.py --strict

# Pull eval data via DVC
dvc pull data/styles.dvc data/artbench10.dvc data/coco_prompts.txt.dvc

# 1. Download generated images from S3
mkdir -p "results/generated/${EXPERIMENT_NAME}"
s3cmd sync -v "${GENERATED_OUTPUT_S3_PATH%/}/" "results/generated/${EXPERIMENT_NAME}/"

# 2. Determine style refs dir from experiment config YAML (folder_path field)
STYLE_REFS_DIR="null"
EXP_CONFIG="/root/b-lora-flux/configs/experiments/${EXPERIMENT_NAME}.yaml"

if [ -f "$EXP_CONFIG" ]; then
  FOLDER_PATH=$(grep -m1 "folder_path:" "$EXP_CONFIG" | sed 's/.*folder_path: "\(.*\)".*/\1/' | tr -d ' ')
  if echo "$FOLDER_PATH" | grep -q "van_gogh"; then
    STYLE_REFS_DIR="data/styles/van_gogh"
  elif echo "$FOLDER_PATH" | grep -q "monet"; then
    STYLE_REFS_DIR="data/styles/monet"
  else
    # No folder_path (inference-only / no-training baseline) — fall back to name, then default
    if echo "$EXPERIMENT_NAME" | grep -q "monet"; then
      STYLE_REFS_DIR="data/styles/monet"
    else
      # Covers e00 baseline, g0x alpha ablations, gs0x split-alpha — all use van_gogh refs
      STYLE_REFS_DIR="data/styles/van_gogh"
    fi
  fi
else
  echo "WARNING: Config not found at $EXP_CONFIG, cannot determine style refs"
fi

echo "Style refs dir: $STYLE_REFS_DIR"

# 3. Compute metrics
python scripts/eval/compute_metrics.py \
  metrics.generated_dir="results/generated/${EXPERIMENT_NAME}" \
  metrics.style_refs_dir="$STYLE_REFS_DIR" \
  metrics.prompt_file=data/coco_prompts.txt \
  metrics.artbench_dir=data/artbench10 \
  metrics.exp_name="$EXPERIMENT_NAME"

# 4. Upload metrics.json to S3
METRICS_JSON="output/results/${EXPERIMENT_NAME}.json"
if [ ! -f "$METRICS_JSON" ]; then
  echo "ERROR: metrics JSON not found at $METRICS_JSON"
  exit 1
fi
s3cmd put -v "$METRICS_JSON" "${METRICS_OUTPUT_S3_PATH%/}/${EXPERIMENT_NAME}.json"
