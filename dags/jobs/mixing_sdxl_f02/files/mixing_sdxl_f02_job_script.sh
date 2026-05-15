#!/bin/bash
set -e

# Env vars injected by Airflow:
#   RUN_TS               — current run's timestamp (e.g. 20260515T214500); subdir under exp_logs/
#   STYLE_VG_RUN_TS      — run_ts of Phase 4.4 SDXL van_gogh training run (default 20260515T083526)
#   STYLE_MONET_RUN_TS   — run_ts where this run's e04_blora_sdxl_monet_img[1-4] LoRAs were written
#   CONTENT_RUN_TS       — run_ts where this run's m02_content_sdxl_<subject> LoRAs were written
#   S3_BASE_PATH         — s3://<bucket>/.../diploma  (used to build per-experiment loras/ paths)
#   MIXING_OUTPUT_S3_PATH — s3://.../exp_logs/${RUN_TS}/f02_blora_sdxl_pairs
#   PAIR_SUBSET          — optional; empty → all 50 pairs; "user_study" → 30-pair subset
#   EXP_NAME             — defaults to f02_blora_sdxl_pairs

EXP_NAME="${EXP_NAME:-f02_blora_sdxl_pairs}"

# 0. Verify Python environment
cd /root/b-lora-flux
python scripts/check_env.py --strict

# 1. Pull eval data (style refs, content refs, prompts manifest is already in repo)
dvc pull data/styles.dvc data/eval_content.dvc data/coco_prompts.txt.dvc

# 2. Stage 8 style LoRAs into input_loras/style/
mkdir -p input_loras/style input_loras/content

BASE="${S3_BASE_PATH%/}"

stage_lora () {
  local s3_dir="$1"   # full s3 path to a per-exp loras dir
  local dst_dir="$2"  # local dir
  local fname="$3"    # final filename (matches mixing_sdxl.yaml id-to-path map)
  local tmp
  tmp=$(mktemp -d)
  echo "Staging ${fname} from ${s3_dir}"
  s3cmd sync -v "${s3_dir%/}/" "${tmp}/"
  local found
  found=$(find "${tmp}" -name "*.safetensors" | head -1)
  if [ -z "${found}" ]; then
    echo "ERROR: no .safetensors found under ${s3_dir}"
    rm -rf "${tmp}"
    exit 1
  fi
  cp "${found}" "${dst_dir}/${fname}"
  rm -rf "${tmp}"
}

# 2a. Van Gogh style (Phase 4.4 — different run_ts than the current F02 run)
for i in 1 2 3 4; do
  stage_lora \
    "${BASE}/exp_logs/${STYLE_VG_RUN_TS}/e04_blora_sdxl_van_gogh_img${i}/loras" \
    "input_loras/style" \
    "e04_blora_sdxl_van_gogh_img${i}.safetensors"
done

# 2b. Monet style (this F02 run)
for i in 1 2 3 4; do
  stage_lora \
    "${BASE}/exp_logs/${STYLE_MONET_RUN_TS}/e04_blora_sdxl_monet_img${i}/loras" \
    "input_loras/style" \
    "e04_blora_sdxl_monet_img${i}.safetensors"
done

# 3. Stage 8 content LoRAs from this F02 run
for subj in backpack bear bowl can cat clock dog vase; do
  stage_lora \
    "${BASE}/exp_logs/${CONTENT_RUN_TS}/m02_content_sdxl_${subj}/loras" \
    "input_loras/content" \
    "m02_content_sdxl_${subj}.safetensors"
done

echo "Staged LoRAs:"
ls -lh input_loras/style input_loras/content

# 4. Build pair_subset override (empty / literal '""' -> null)
# Airflow may serialize an empty string param as the 2-char literal '""',
# so strip surrounding quotes before checking emptiness.
_PS="${PAIR_SUBSET:-}"
_PS="${_PS#\"}"; _PS="${_PS%\"}"
_PS="${_PS#\'}"; _PS="${_PS%\'}"
if [ -z "${_PS}" ]; then
  PAIR_SUBSET_OVR="mixing_sdxl.pair_subset=null"
  METRICS_PAIR_SUBSET_OVR="f02_metrics.pair_subset=null"
else
  PAIR_SUBSET_OVR="mixing_sdxl.pair_subset=${_PS}"
  METRICS_PAIR_SUBSET_OVR="f02_metrics.pair_subset=${_PS}"
fi

# 5. Generate 50 (or 30) pair images
mkdir -p output/mixing_sdxl
python scripts/eval/generate_mixing_sdxl.py \
  mixing_sdxl.manifest_path=experiments/data/b_lora_eval_pairs.json \
  mixing_sdxl.style_lora_dir=input_loras/style \
  mixing_sdxl.content_lora_dir=input_loras/content \
  mixing_sdxl.output_dir=output/mixing_sdxl \
  mixing_sdxl.exp_name="${EXP_NAME}" \
  "${PAIR_SUBSET_OVR}"

# 6. Compute F02 metrics
python scripts/eval/compute_f02_metrics.py \
  f02_metrics.manifest_path=experiments/data/b_lora_eval_pairs.json \
  f02_metrics.generated_dir="output/mixing_sdxl/${EXP_NAME}" \
  f02_metrics.exp_name="${EXP_NAME}" \
  "${METRICS_PAIR_SUBSET_OVR}"

# 7. Upload generated images + metrics to S3
METRICS_JSON="output/results/${EXP_NAME}_f02_metrics.json"
if [ ! -f "${METRICS_JSON}" ]; then
  echo "ERROR: metrics JSON not found at ${METRICS_JSON}"
  exit 1
fi

s3cmd sync -v --follow-symlinks "output/mixing_sdxl/${EXP_NAME}/" "${MIXING_OUTPUT_S3_PATH%/}/generated/"
s3cmd put -v "${METRICS_JSON}" "${MIXING_OUTPUT_S3_PATH%/}/${EXP_NAME}_f02_metrics.json"

echo "F02 mixing-batch done. Output at ${MIXING_OUTPUT_S3_PATH%/}/"
