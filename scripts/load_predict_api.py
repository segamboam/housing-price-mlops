#!/usr/bin/env python3
"""Generate load on /predict and /predict/batch for Grafana demo.

Runs for a configurable duration at a configurable interval. Payloads are
randomly varied within valid ranges to simulate more realistic traffic.

Usage:
  uv run python scripts/load_predict_api.py
  DURATION=30 INTERVAL=0.5 uv run python scripts/load_predict_api.py
  make load-demo DURATION=60 INTERVAL=1
"""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    return float(v)


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    return int(v)


# -----------------------------------------------------------------------------
# Config (overridable via env and make)
# -----------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY")
DURATION_SEC = _env_int("DURATION", 15)
INTERVAL_SEC = _env_float("INTERVAL", 1.0)

# Pydantic valid ranges (schema) for clamping
RANGES = {
    "CRIM": (0, None),
    "ZN": (0, 100),
    "INDUS": (0, 100),
    "CHAS": (0, 1),
    "NOX": (0, 1),
    "RM": (1, 15),
    "AGE": (0, 100),
    "DIS": (0.01, None),
    "RAD": (1, 24),
    "TAX": (0, None),
    "PTRATIO": (0, None),
    "B": (0, 400),
    "LSTAT": (0, 100),
}


def _clamp(key: str, value: float) -> float:
    lo, hi = RANGES.get(key, (None, None))
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


def _perturb(payload: dict, jitter: float = 0.2) -> dict:
    """Return a copy of payload with values randomly varied within valid ranges."""
    out = {}
    for k, v in payload.items():
        if isinstance(v, int) and k != "CHAS":
            # RAD: integer
            delta = random.uniform(-jitter * v, jitter * v) if v else 0
            out[k] = int(round(_clamp(k, v + delta)))
        elif isinstance(v, float):
            delta = random.uniform(-jitter * abs(v), jitter * abs(v)) if v else random.uniform(-jitter, jitter)
            out[k] = round(_clamp(k, v + delta), 4)
        else:
            out[k] = v
    return out

# -----------------------------------------------------------------------------
# Payloads: different conditions for single predict
# -----------------------------------------------------------------------------
# In-range (typical Boston Housing)
PAYLOAD_NORMAL_1 = {
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98,
}
PAYLOAD_NORMAL_2 = {
    "CRIM": 0.02731,
    "ZN": 0.0,
    "INDUS": 7.07,
    "CHAS": 0,
    "NOX": 0.469,
    "RM": 6.421,
    "AGE": 78.9,
    "DIS": 4.9671,
    "RAD": 2,
    "TAX": 242.0,
    "PTRATIO": 17.8,
    "B": 396.9,
    "LSTAT": 9.14,
}
PAYLOAD_NORMAL_3 = {
    "CRIM": 0.06905,
    "ZN": 0.0,
    "INDUS": 2.18,
    "CHAS": 0,
    "NOX": 0.458,
    "RM": 7.147,
    "AGE": 54.2,
    "DIS": 6.0622,
    "RAD": 3,
    "TAX": 222.0,
    "PTRATIO": 18.7,
    "B": 396.9,
    "LSTAT": 5.33,
}

# Out-of-range vs typical training data (to trigger drift / warnings in Grafana)
# LSTAT and RM often outside training min/max
PAYLOAD_DRIFT_1 = {
    "CRIM": 0.1,
    "ZN": 20.0,
    "INDUS": 5.0,
    "CHAS": 0,
    "NOX": 0.5,
    "RM": 2.5,   # low vs typical 3.5–8.8
    "AGE": 50.0,
    "DIS": 5.0,
    "RAD": 5,
    "TAX": 300.0,
    "PTRATIO": 14.0,
    "B": 350.0,
    "LSTAT": 50.0,  # high vs typical 1.7–38
}
PAYLOAD_DRIFT_2 = {
    "CRIM": 90.0,   # very high vs typical 0–89
    "ZN": 10.0,
    "INDUS": 20.0,
    "CHAS": 1,
    "NOX": 0.8,
    "RM": 9.0,
    "AGE": 95.0,
    "DIS": 1.5,
    "RAD": 24,
    "TAX": 700.0,
    "PTRATIO": 22.0,
    "B": 100.0,
    "LSTAT": 35.0,
}

SINGLE_PAYLOADS = [
    PAYLOAD_NORMAL_1,
    PAYLOAD_NORMAL_2,
    PAYLOAD_DRIFT_1,
    PAYLOAD_NORMAL_3,
    PAYLOAD_DRIFT_2,
]

# Batch templates (items will be perturbed when building request)
BATCH_TEMPLATES = [
    [PAYLOAD_NORMAL_1, PAYLOAD_NORMAL_2],
    [PAYLOAD_NORMAL_1, PAYLOAD_DRIFT_1, PAYLOAD_NORMAL_3],
    [PAYLOAD_NORMAL_2, PAYLOAD_DRIFT_2],
]


def _post(url: str, payload: dict, headers: dict, timeout: int = 10) -> tuple[int, dict | None]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200] if e.fp else ""
        return e.code, {"_raw": body}
    except Exception:
        raise


def main() -> None:
    base = API_BASE_URL.rstrip("/")
    predict_url = f"{base}/predict"
    batch_url = f"{base}/predict/batch"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    print(f"Target: {API_BASE_URL} (duration={DURATION_SEC}s, interval={INTERVAL_SEC}s)")
    if API_KEY:
        print("Using X-API-Key")
    print()

    start = time.monotonic()
    count_predict = 0
    count_batch = 0
    errors = 0
    n_single = len(SINGLE_PAYLOADS)
    n_batch = len(BATCH_TEMPLATES)

    while (time.monotonic() - start) < DURATION_SEC:
        elapsed = time.monotonic() - start
        step = int(elapsed / INTERVAL_SEC)
        if step % 2 == 0:
            base = SINGLE_PAYLOADS[step % n_single]
            payload = _perturb(base, jitter=0.15)
            try:
                status, body = _post(predict_url, payload, headers, timeout=5)
                if 200 <= status < 300 and isinstance(body, dict):
                    count_predict += 1
                    pred = body.get("prediction", "?")
                    w = body.get("warnings", [])
                    print(f"  [{elapsed:5.1f}s] POST /predict -> {status} pred={pred} warnings={len(w)}")
                else:
                    errors += 1
                    raw = body.get("_raw", str(body)) if isinstance(body, dict) else str(body)
                    print(f"  [{elapsed:5.1f}s] POST /predict -> {status} {raw[:80]}")
            except Exception as e:
                errors += 1
                print(f"  [{elapsed:5.1f}s] POST /predict error: {e}")
        else:
            items = [_perturb(p, jitter=0.15) for p in BATCH_TEMPLATES[step % n_batch]]
            payload = {"items": items}
            try:
                status, body = _post(batch_url, payload, headers, timeout=10)
                if 200 <= status < 300 and isinstance(body, dict):
                    count_batch += 1
                    n_items = body.get("total_items", 0)
                    print(f"  [{elapsed:5.1f}s] POST /predict/batch -> {status} items={n_items}")
                else:
                    errors += 1
                    raw = body.get("_raw", str(body)) if isinstance(body, dict) else str(body)
                    print(f"  [{elapsed:5.1f}s] POST /predict/batch -> {status} {raw[:80]}")
            except Exception as e:
                errors += 1
                print(f"  [{elapsed:5.1f}s] POST /predict/batch error: {e}")

        time.sleep(max(0, INTERVAL_SEC - (time.monotonic() - start - elapsed)))

    total = count_predict + count_batch
    print()
    print(f"Done. Requests: {total} (predict={count_predict}, batch={count_batch}), errors={errors}")
    print("Check Grafana (Housing Price API dashboard) for updated metrics.")


if __name__ == "__main__":
    main()
