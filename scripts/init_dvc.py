#!/usr/bin/env python
"""Initialize DVC for data versioning.

This script ensures DVC is properly configured for preprocessing cache.
- Raw data (HousingData.csv) is kept in Git for easy evaluation
- Preprocessed data is cached locally and synced to MinIO via DVC

Usage:
    uv run python scripts/init_dvc.py

    # Or via Makefile:
    make dvc-init
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    if not quiet:
        print(f"  Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def is_dvc_initialized() -> bool:
    """Check if DVC is already initialized in this repo."""
    dvc_dir = Path(".dvc")
    return dvc_dir.exists() and (dvc_dir / "config").exists()


def is_data_available() -> bool:
    """Check if the raw data file exists."""
    return Path("data/HousingData.csv").exists()


def init_dvc() -> None:
    """Initialize DVC for preprocessing cache."""
    print("\n=== DVC Initialization ===\n")

    # Check if DVC is installed
    result = run_command(["dvc", "version"], check=False, quiet=True)
    if result.returncode != 0:
        print("ERROR: DVC is not installed. Run: uv sync")
        sys.exit(1)
    version = result.stdout.strip().split('\n')[0] if result.stdout else 'unknown'
    print(f"  DVC version: {version}")

    # Step 1: Initialize DVC
    if is_dvc_initialized():
        print("\n[1/3] DVC already initialized")
    else:
        print("\n[1/3] Initializing DVC...")
        result = run_command(["dvc", "init"], check=False)
        if result.returncode != 0:
            print(f"  Warning: {result.stderr}")
        else:
            print("  DVC initialized successfully")

    # Step 2: Verify/create config
    config_path = Path(".dvc/config")
    if config_path.exists():
        print("\n[2/3] DVC config exists")
    else:
        print("\n[2/3] Creating DVC config...")
        config_content = """[core]
    remote = minio
    autostage = true
[remote "minio"]
    url = s3://dvc-artifacts
    endpointurl = http://localhost:9000
    access_key_id = minioadmin
    secret_access_key = minioadmin123
"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content)
        print("  Config created with MinIO remote")

    # Step 3: Verify raw data exists
    if not is_data_available():
        print(f"\n[3/3] ERROR: Raw data not found: data/HousingData.csv")
        print("  The raw data should be in the Git repository.")
        sys.exit(1)
    print(f"\n[3/3] Raw data available: data/HousingData.csv")

    # Create processed directory
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("DVC SETUP COMPLETE")
    print("=" * 60)
    print(f"  DVC initialized:    {is_dvc_initialized()}")
    print(f"  Raw data in Git:    {is_data_available()}")
    print(f"  Remote:             MinIO (s3://dvc-artifacts)")
    print("")
    print("  Preprocessing cache:")
    print("    - Managed automatically by make train/experiment")
    print("    - Stored in data/processed/{version}/")
    print("    - Synced to MinIO via DVC")
    print("")
    print("  Commands:")
    print("    make train       # Interactive training")
    print("    make experiment  # Grid search (config.yaml)")
    print("    make cache-status # Show cached preprocessings")
    print("=" * 60)


if __name__ == "__main__":
    init_dvc()
