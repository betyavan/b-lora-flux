#!/bin/bash
set -e

# 0. Verify Python environment before starting expensive training
cd /root/b-lora-flux
python scripts/check_env.py --strict

# 1. Run ai-toolkit training
cd /root/b-lora-flux
python src/ai-toolkit/run.py configs/experiments/${EXPERIMENT_NAME}.yaml

# 3. Collect .safetensors to results/loras/
mkdir -p results/loras
find output/ -name "*.safetensors" -exec cp {} results/loras/ \;
echo $(date -u +%Y-%m-%dT%H:%M:%SZ) > results/loras/train_done.txt

# 4. Upload to S3 via s3cmd
s3cmd sync -v --follow-symlinks results/loras/ ${TRAIN_OUTPUT_S3_PATH%/}/
