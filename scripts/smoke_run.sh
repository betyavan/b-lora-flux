#!/usr/bin/env bash
# smoke_run.sh — Run the full pipeline with the minimal smoke_test config
# and verify expected S3 artifacts are produced.
set -e

# ---------------------------------------------------------------------------
# Source corporate environment variables
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
ENV_FILE="${ROOT_DIR}/infra.env"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo "[warn] ${ENV_FILE} not found — relying on existing environment"
fi

# ---------------------------------------------------------------------------
# Timeout guard (30 minutes total)
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS=1800
DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))

ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

elapsed_ok() {
    [[ $(date +%s) -lt ${DEADLINE} ]]
}

echo "[$(ts)] smoke_run.sh starting (timeout: ${TIMEOUT_SECONDS}s)"

# ---------------------------------------------------------------------------
# Step 1 — Pre-flight connectivity checks
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] Step 1: Running connectivity checks..."
poetry run python "${SCRIPT_DIR}/check_connectivity.py"
echo "[$(ts)] Connectivity checks passed."

# ---------------------------------------------------------------------------
# Step 2 — Trigger the DAG
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] Step 2: Triggering blora_flux_pipeline with EXP=smoke_test..."
airflow dags trigger blora_flux_pipeline \
    --conf "{\"EXPERIMENT_NAME\": \"smoke_test\", \"S3_BASE_PATH\": \"${CORP_S3_BASE_PATH}\"}"
echo "[$(ts)] DAG triggered."

# Give Airflow a moment to register the run
sleep 5

# ---------------------------------------------------------------------------
# Step 3 — Poll until the run leaves the running state
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] Step 3: Polling for DAG completion (every 30s, up to ${TIMEOUT_SECONDS}s total)..."

while true; do
    if ! elapsed_ok; then
        echo "[$(ts)] TIMEOUT: smoke run exceeded ${TIMEOUT_SECONDS}s — aborting."
        exit 1
    fi

    RUNNING=$(airflow dags list-runs -d blora_flux_pipeline --state running 2>/dev/null | grep -c "smoke_test" || true)

    if [[ "${RUNNING}" -eq 0 ]]; then
        echo "[$(ts)] DAG run is no longer in running state."
        break
    fi

    echo "[$(ts)]   ... still running (${RUNNING} active run(s)). Sleeping 30s..."
    sleep 30
done

# ---------------------------------------------------------------------------
# Step 4 — Verify expected S3 artifacts
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] Step 4: Checking S3 artifacts..."

S3_ENDPOINT="${CORP_S3_ENDPOINT:-}"
S3_BUCKET="${CORP_S3_BUCKET_MODELS:-}"
BASE_PATH="${CORP_S3_BASE_PATH:-}"

# Strip leading s3://bucket/ from BASE_PATH if present
if [[ "${BASE_PATH}" == s3://* ]]; then
    _stripped="${BASE_PATH#s3://}"
    S3_BUCKET="${_stripped%%/*}"
    BASE_PATH="${_stripped#*/}"
fi

LOGS_PREFIX="${BASE_PATH%/}/exp_logs/smoke_test"

check_s3_prefix_exists() {
    local desc="$1"
    local prefix="$2"
    local count
    count=$(aws s3 ls "s3://${S3_BUCKET}/${prefix}" \
        --endpoint-url "${S3_ENDPOINT}" \
        --recursive 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    if [[ "${count}" -gt 0 ]]; then
        echo "  ✓ PASS  ${desc} (${count} object(s))"
        return 0
    else
        echo "  ✗ FAIL  ${desc} — nothing found at s3://${S3_BUCKET}/${prefix}"
        return 1
    fi
}

ARTIFACT_FAILURES=0

# 4a — LoRA weights (.safetensors)
check_s3_prefix_exists \
    "LoRA weights (.safetensors)" \
    "${LOGS_PREFIX}/loras/" || (( ARTIFACT_FAILURES++ )) || true

# 4b — Generated images directory
check_s3_prefix_exists \
    "Generated images" \
    "${LOGS_PREFIX}/generated/" || (( ARTIFACT_FAILURES++ )) || true

# 4c — Metrics JSON
check_s3_prefix_exists \
    "Metrics JSON" \
    "${LOGS_PREFIX}/metrics/metrics.json" || (( ARTIFACT_FAILURES++ )) || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] ----------------------------------------"
if [[ "${ARTIFACT_FAILURES}" -eq 0 ]]; then
    echo "[$(ts)] SMOKE TEST PASSED — all 3 artifacts found."
    exit 0
else
    echo "[$(ts)] SMOKE TEST FAILED — ${ARTIFACT_FAILURES} artifact(s) missing."
    exit 1
fi
