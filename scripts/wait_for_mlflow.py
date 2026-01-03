#!/usr/bin/env python
"""Wait for MLflow to be healthy before proceeding.

Cross-platform script that polls MLflow health endpoint.

Usage:
    uv run python scripts/wait_for_mlflow.py [--timeout 120]
"""

import argparse
import sys
import time
import urllib.error
import urllib.request


def wait_for_mlflow(url: str = "http://localhost:5000/health", timeout: int = 120) -> bool:
    """Wait for MLflow to respond to health checks.

    Args:
        url: MLflow health endpoint URL
        timeout: Maximum time to wait in seconds

    Returns:
        True if MLflow is healthy, False if timeout reached
    """
    print(f"Waiting for MLflow at {url} (timeout: {timeout}s)...")
    start_time = time.time()
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    elapsed = time.time() - start_time
                    print(f"MLflow is healthy! (took {elapsed:.1f}s, {attempt} attempts)")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        # Progress indicator
        if attempt % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Still waiting... ({elapsed:.0f}s elapsed)")

        time.sleep(2)

    print(f"ERROR: MLflow did not become healthy within {timeout}s")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for MLflow to be healthy")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    parser.add_argument("--url", default="http://localhost:5000/health", help="MLflow health URL")
    args = parser.parse_args()

    if wait_for_mlflow(url=args.url, timeout=args.timeout):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
