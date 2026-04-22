#!/usr/bin/env python3
"""Pre-flight connectivity checks for the B-LoRA FLUX pipeline.

Checks S3, Airflow CLI, mlc CLI, data path, and output path writability.
Loads environment variables from infra.env at startup.
Exit code 0 if all checks pass, 1 if any fail.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Load infra.env at startup (python-dotenv if available, else manual parse)
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent.parent / "infra.env"


def _load_env(path: Path) -> None:
    if not path.exists():
        print(f"  [warn] {path} not found — relying on existing environment variables")
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import]

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    # Manual parse: KEY=VALUE lines, skip comments and blanks
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env(_ENV_FILE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "✓"  # ✓
FAIL = "✗"  # ✗

_results: list[tuple[str, bool]] = []


def _record(label: str, ok: bool, detail: str = "") -> None:
    symbol = PASS if ok else FAIL
    msg = f"  {symbol} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    _results.append((label, ok))


# ---------------------------------------------------------------------------
# Check 1 — S3 endpoint reachable
# ---------------------------------------------------------------------------
def check_s3_endpoint() -> None:
    label = "S3 endpoint reachable"
    endpoint = os.environ.get("CORP_S3_ENDPOINT", "")
    if not endpoint:
        _record(label, False, "CORP_S3_ENDPOINT not set")
        return
    try:
        import boto3  # type: ignore[import]

        client = boto3.client("s3", endpoint_url=endpoint)
        client.list_buckets()
        _record(label, True, endpoint)
    except Exception as exc:  # noqa: BLE001
        _record(label, False, str(exc)[:120])


# ---------------------------------------------------------------------------
# Check 2 — Airflow CLI and expected DAGs
# ---------------------------------------------------------------------------
_EXPECTED_DAGS = {"blora_flux_pipeline", "blora_flux_group_pipeline"}


def check_airflow() -> None:
    label = "Airflow CLI"
    try:
        result = subprocess.run(
            ["airflow", "dags", "list"],
            capture_output=True,
            timeout=10,
            text=True,
        )
        if result.returncode != 0:
            _record(label, False, result.stderr.strip()[:120] or "non-zero exit")
            return
        output = result.stdout + result.stderr
        missing = [d for d in _EXPECTED_DAGS if d not in output]
        if missing:
            _record(label, False, f"DAGs not found: {', '.join(missing)}")
        else:
            _record(label, True, f"DAGs present: {', '.join(_EXPECTED_DAGS)}")
    except FileNotFoundError:
        _record(label, False, "airflow binary not found")
    except subprocess.TimeoutExpired:
        _record(label, False, "timed out after 10 s")
    except Exception as exc:  # noqa: BLE001
        _record(label, False, str(exc)[:120])


# ---------------------------------------------------------------------------
# Check 3 — mlc CLI
# ---------------------------------------------------------------------------
def check_mlc() -> None:
    label = "mlc CLI"
    try:
        result = subprocess.run(
            ["mlc", "--version"],
            capture_output=True,
            timeout=10,
            text=True,
        )
        if result.returncode == 0:
            version = (result.stdout + result.stderr).strip().splitlines()[0]
            _record(label, True, version)
        else:
            _record(label, False, result.stderr.strip()[:120] or "non-zero exit")
    except FileNotFoundError:
        _record(label, False, "mlc binary not found")
    except subprocess.TimeoutExpired:
        _record(label, False, "timed out after 10 s")
    except Exception as exc:  # noqa: BLE001
        _record(label, False, str(exc)[:120])


# ---------------------------------------------------------------------------
# Check 4 — S3 data path readable
# ---------------------------------------------------------------------------
def check_s3_data_path() -> None:
    label = "S3 data path readable"
    endpoint = os.environ.get("CORP_S3_ENDPOINT", "")
    bucket = os.environ.get("CORP_S3_BUCKET_DATA", "")
    prefix = os.environ.get("CORP_S3_DATA_PATH", "")
    if not all([endpoint, bucket, prefix]):
        _record(
            label,
            False,
            "one of CORP_S3_ENDPOINT / CORP_S3_BUCKET_DATA / CORP_S3_DATA_PATH not set",
        )
        return
    try:
        import boto3  # type: ignore[import]

        client = boto3.client("s3", endpoint_url=endpoint)
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        count = resp.get("KeyCount", 0)
        _record(label, True, f"s3://{bucket}/{prefix} ({count} object(s) found)")
    except Exception as exc:  # noqa: BLE001
        _record(label, False, str(exc)[:120])


# ---------------------------------------------------------------------------
# Check 5 — S3 output path writable
# ---------------------------------------------------------------------------
def check_s3_output_writable() -> None:
    label = "S3 output path writable"
    endpoint = os.environ.get("CORP_S3_ENDPOINT", "")
    base_path = os.environ.get("CORP_S3_BASE_PATH", "")
    # Derive bucket and key prefix from CORP_S3_BASE_PATH (may be s3://bucket/key)
    bucket_models = os.environ.get("CORP_S3_BUCKET_MODELS", "")
    if not all([endpoint, base_path, bucket_models]):
        _record(
            label,
            False,
            "one of CORP_S3_ENDPOINT / CORP_S3_BASE_PATH / CORP_S3_BUCKET_MODELS not set",
        )
        return
    # Strip s3://bucket/ prefix if present, leaving the key prefix
    key_prefix = base_path
    if key_prefix.startswith("s3://"):
        parts = key_prefix[5:].split("/", 1)
        bucket_models = parts[0]
        key_prefix = parts[1] if len(parts) > 1 else ""
    marker_key = f"{key_prefix.rstrip('/')}/smoke_test/.connectivity_check_{uuid.uuid4().hex}"
    try:
        import boto3  # type: ignore[import]

        client = boto3.client("s3", endpoint_url=endpoint)
        client.put_object(Bucket=bucket_models, Key=marker_key, Body=b"connectivity-check")
        client.delete_object(Bucket=bucket_models, Key=marker_key)
        _record(label, True, f"s3://{bucket_models}/{key_prefix}")
    except Exception as exc:  # noqa: BLE001
        _record(label, False, str(exc)[:120])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Running pre-flight connectivity checks...\n")
    check_s3_endpoint()
    check_airflow()
    check_mlc()
    check_s3_data_path()
    check_s3_output_writable()

    passed = sum(1 for _, ok in _results if ok)
    total = len(_results)
    print(f"\nSummary: {passed}/{total} checks passed")
    if passed < total:
        failed = [label for label, ok in _results if not ok]
        print(f"Failed : {', '.join(failed)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
