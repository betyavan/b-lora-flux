#!/usr/bin/env bash
# smoke_run.sh — Trigger the minimal smoke_test pipeline and verify S3 artifacts.
# Run this from the project root on a machine where `airflow` and `aws` CLIs are available.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
ENV_FILE="${ROOT_DIR}/infra.env"

if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
else
    echo "[warn] infra.env not found — relying on existing environment"
fi

TIMEOUT_SECONDS=1800
DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))
ts() { date '+%Y-%m-%d %H:%M:%S'; }
elapsed_ok() { [[ $(date +%s) -lt ${DEADLINE} ]]; }

echo "[$(ts)] smoke_run.sh starting (timeout: ${TIMEOUT_SECONDS}s)"

# ---------------------------------------------------------------------------
# Step 1 — Trigger the DAG
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] Step 1: Triggering blora_flux_pipeline with EXP=smoke_test..."
airflow dags trigger blora_flux_pipeline \
    --conf "{\"EXPERIMENT_NAME\": \"smoke_test\", \"S3_BASE_PATH\": \"${CORP_S3_BASE_PATH}\"}"
echo "[$(ts)] DAG triggered."
sleep 5

# ---------------------------------------------------------------------------
# Step 2 — Poll until the run leaves the running state
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] Step 2: Polling for DAG completion (every 30s, up to ${TIMEOUT_SECONDS}s)..."

while true; do
    if ! elapsed_ok; then
        echo "[$(ts)] TIMEOUT: smoke run exceeded ${TIMEOUT_SECONDS}s — aborting."
        exit 1
    fi

    RUNNING=$(airflow dags list-runs -d blora_flux_pipeline --state running 2>/dev/null \
        | grep -c "smoke_test" || true)

    if [[ "${RUNNING}" -eq 0 ]]; then
        echo "[$(ts)] DAG run finished."
        break
    fi

    echo "[$(ts)]   ... still running (${RUNNING} active). Sleeping 30s..."
    sleep 30
done

# ---------------------------------------------------------------------------
# Step 3 — Verify S3 artifacts
# ---------------------------------------------------------------------------
echo ""
echo "[$(ts)] Step 3: Checking S3 artifacts..."

BASE_PATH="${CORP_S3_BASE_PATH:-}"
S3_ENDPOINT="${CORP_S3_ENDPOINT:-}"

# Strip s3://bucket/ prefix to get bucket and path separately
if [[ "${BASE_PATH}" == s3://* ]]; then
    _stripped="${BASE_PATH#s3://}"
    S3_BUCKET="${_stripped%%/*}"
    BASE_KEY="${_stripped#*/}"
else
    echo "[error] CORP_S3_BASE_PATH must start with s3://"
    exit 1
fi

LOGS_PREFIX="${BASE_KEY%/}/exp_logs/smoke_test"

check_s3() {
    local desc="$1" prefix="$2" count
    count=$(aws s3 ls "s3://${S3_BUCKET}/${prefix}" \
        --endpoint-url "${S3_ENDPOINT}" --recursive 2>/dev/null | wc -l | tr -d ' ' || echo "0")
    if [[ "${count}" -gt 0 ]]; then
        echo "  ✓ PASS  ${desc} (${count} object(s))"
    else
        echo "  ✗ FAIL  ${desc} — nothing at s3://${S3_BUCKET}/${prefix}"
        return 1
    fi
}

FAILS=0
check_s3 "LoRA weights"     "${LOGS_PREFIX}/loras/"          || (( FAILS++ )) || true
check_s3 "Generated images" "${LOGS_PREFIX}/generated/"       || (( FAILS++ )) || true
check_s3 "metrics.json"     "${LOGS_PREFIX}/metrics/metrics.json" || (( FAILS++ )) || true

echo ""
echo "[$(ts)] ----------------------------------------"
if [[ "${FAILS}" -eq 0 ]]; then
    echo "[$(ts)] SMOKE TEST PASSED — all 3 artifacts found in S3."
    exit 0
else
    echo "[$(ts)] SMOKE TEST FAILED — ${FAILS} artifact(s) missing."
    exit 1
fi
