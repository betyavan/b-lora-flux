#!/bin/bash
set -e

# 1. Set up data symlinks (data/ -> /my_datasets/)
mkdir -p /root/b-lora-flux/data
ln -sfn /my_datasets/styles /root/b-lora-flux/data/styles
ln -sfn /my_datasets/artbench10 /root/b-lora-flux/data/artbench10
ln -sfn /my_datasets/coco_prompts.txt /root/b-lora-flux/data/coco_prompts.txt

# 2. Run ai-toolkit training
cd /root/b-lora-flux
python run.py configs/experiments/${EXPERIMENT_NAME}.yaml

# 3. Collect .safetensors to results/loras/
mkdir -p results/loras
find output/ -name "*.safetensors" -exec cp {} results/loras/ \;
echo $(date -u +%Y-%m-%dT%H:%M:%SZ) > results/loras/train_done.txt

# 4. Upload to S3 via s3cmd
s3cmd sync -v --follow-symlinks results/loras/ ${TRAIN_OUTPUT_S3_PATH%/}/
