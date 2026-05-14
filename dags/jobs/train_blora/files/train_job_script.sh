#!/bin/bash
set -e

# 0. Verify Python environment before starting expensive training
cd /root/b-lora-flux
python scripts/check_env.py --strict

# 1a. Pull training data via DVC
dvc pull data/styles.dvc

# 1. Check if experiment has a training process (baseline configs have process: [])
PROCESS_COUNT=$(python3 -c "
import yaml, sys
with open('configs/experiments/${EXPERIMENT_NAME}.yaml') as f:
    cfg = yaml.safe_load(f)
process = cfg.get('config', {}).get('process', [])
print(len(process))
")

if [ "$PROCESS_COUNT" -eq 0 ]; then
  mkdir -p results/loras
  if [ -n "${PHASE3_BASE_LORA_S3_PATH:-}" ]; then
    echo "INFO: ${EXPERIMENT_NAME} — Phase 3 mode: copying base LoRA from ${PHASE3_BASE_LORA_S3_PATH}"
    s3cmd sync -v "${PHASE3_BASE_LORA_S3_PATH%/}/" results/loras/
  else
    echo "INFO: ${EXPERIMENT_NAME} has no training process — skipping training (baseline mode)"
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) baseline_no_training" > results/loras/train_done.txt
  s3cmd sync -v --follow-symlinks results/loras/ "${TRAIN_OUTPUT_S3_PATH%/}/"
  exit 0
fi

# 2. Run ai-toolkit training
cd /root/b-lora-flux
python src/ai-toolkit/run.py configs/experiments/${EXPERIMENT_NAME}.yaml

# 3. Collect .safetensors to results/loras/  (skipped for baseline — handled above)
mkdir -p results/loras
find output/ -name "*.safetensors" -exec cp {} results/loras/ \;
echo $(date -u +%Y-%m-%dT%H:%M:%SZ) > results/loras/train_done.txt

# 4. Upload to S3 via s3cmd
s3cmd sync -v --follow-symlinks results/loras/ ${TRAIN_OUTPUT_S3_PATH%/}/
