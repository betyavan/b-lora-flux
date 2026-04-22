#!/usr/bin/env python3
"""Check that all required packages are importable on the remote compute node.

Run this at the start of each H100 job to catch missing deps before
starting an expensive training run.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --strict   # exit 1 on any failure
"""
from __future__ import annotations

import argparse
import importlib
import sys

REQUIRED: list[tuple[str, str]] = [
    # (import_name, pip_name)
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("safetensors", "safetensors"),
    ("omegaconf", "omegaconf"),
    ("hydra", "hydra-core"),
    ("PIL", "Pillow"),
    ("datasets", "datasets"),
    ("torchmetrics", "torchmetrics"),
    ("lpips", "lpips"),
    ("clean_fid", "clean-fid"),
    ("boto3", "boto3"),
    ("s3cmd", None),   # CLI tool, not a Python package — checked separately
]

OPTIONAL: list[tuple[str, str]] = [
    ("clearml", "clearml"),
]


def _check_import(mod: str) -> str | None:
    try:
        importlib.import_module(mod)
        return None
    except ImportError as e:
        return str(e)


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if available else "—"
        return available, f"{count}× {name}" if available else "not available"
    except ImportError:
        return False, "torch not installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any required package is missing")
    args = parser.parse_args()

    print("Checking Python environment on compute node...")
    print(f"Python {sys.version}")
    print()

    failures: list[str] = []

    print("Required packages:")
    for mod, pip in REQUIRED:
        if mod == "s3cmd":
            import subprocess
            ok = subprocess.run(["which", "s3cmd"], capture_output=True).returncode == 0
            status = "✓" if ok else "✗"
            note = "" if ok else f"  ← install: pip install s3cmd"
            print(f"  {status} s3cmd (CLI){note}")
            if not ok:
                failures.append("s3cmd")
            continue

        err = _check_import(mod)
        if err is None:
            # Print version if available
            try:
                ver = importlib.import_module(mod).__version__  # type: ignore[attr-defined]
                print(f"  ✓ {mod}=={ver}")
            except AttributeError:
                print(f"  ✓ {mod}")
        else:
            install = f"pip install {pip}" if pip else mod
            print(f"  ✗ {mod}  ← {install}")
            failures.append(mod)

    print()
    print("Optional packages:")
    for mod, pip in OPTIONAL:
        err = _check_import(mod)
        if err is None:
            try:
                ver = importlib.import_module(mod).__version__  # type: ignore[attr-defined]
                print(f"  ✓ {mod}=={ver}")
            except AttributeError:
                print(f"  ✓ {mod}")
        else:
            print(f"  - {mod}  (not installed, OK)")

    print()
    cuda_ok, cuda_info = _check_cuda()
    print(f"CUDA: {'✓' if cuda_ok else '✗'} {cuda_info}")

    print()
    if failures:
        print(f"MISSING: {', '.join(failures)}")
        if args.strict:
            sys.exit(1)
    else:
        print("All required packages OK.")


if __name__ == "__main__":
    main()
