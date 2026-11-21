"""One-weld, sensor-only forecast comparison in a Modal CPU Sandbox.

Downloads only the pilot CSV through a short-lived signed URL, runs every
cadence mode x model (plus naive baselines) on the steady arc-on window, and
prints one JSON metrics record. It uploads nothing and prints no raw rows,
forecasts or URLs.
"""

import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import numpy as np

from weldfusion.dataset import load_sensor_table
from weldfusion.timestral_sensor_pilot import MODEL_IDS, load_forecaster, run_pilot


MAX_CSV_BYTES = 10 * 1024 * 1024


def download(url, path):
    try:
        with urllib.request.urlopen(url, timeout=60) as source, open(path, "wb") as target:
            size = 0
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_CSV_BYTES:
                    raise RuntimeError("pilot CSV exceeds size limit")
                target.write(block)
    except Exception:
        raise RuntimeError("pilot CSV download failed") from None
    return size


def cadence_summary(timestamps):
    """Gap statistics only; no raw sensor rows or timestamps leave the Sandbox."""
    if timestamps is None:
        return None
    gaps = np.diff(np.asarray(timestamps, dtype=float))
    median = float(np.median(gaps))
    rounded, counts = np.unique(np.round(gaps, 3), return_counts=True)
    top = sorted(zip(counts.tolist(), rounded.tolist()), reverse=True)[:8]
    return {"n_gaps": int(gaps.size), "median_s": median, "min_s": float(gaps.min()),
            "max_s": float(gaps.max()), "p05_s": float(np.percentile(gaps, 5)),
            "p95_s": float(np.percentile(gaps, 95)),
            "n_outside_25pct": int(np.sum(np.abs(gaps - median) / median > 0.25)) if median > 0 else None,
            "n_nonpositive": int(np.sum(gaps <= 0)),
            "top_gap_values_s": [{"gap_s": g, "count": c} for c, g in top]}


def main():
    sample_id = os.environ["PILOT_SAMPLE_ID"]
    columns = json.loads(os.environ["PILOT_COLUMNS"])
    models = json.loads(os.environ.get("PILOT_MODELS", '["timestral", "timesfm"]'))
    cadences = json.loads(os.environ.get("PILOT_CADENCES", '["strict", "rows", "resample"]'))
    dry_run = os.environ.get("PILOT_DRY_RUN") == "1"
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "{}.csv".format(sample_id)
        csv_bytes = download(os.environ.pop("PILOT_GET_CSV"), csv_path)
        table = load_sensor_table(csv_path)
    load_s, forecasters = {}, {}
    if not dry_run:
        for name in models:
            start = time.perf_counter()
            forecasters[name] = load_forecaster(name)
            load_s[name] = round(time.perf_counter() - start, 3)
    runs = run_pilot(table, columns, forecasters, cadences,
                     horizon=int(os.environ["PILOT_HORIZON"]),
                     context=int(os.environ["PILOT_CONTEXT"]),
                     window=os.environ.get("PILOT_WINDOW", "arc_on"))
    print(json.dumps({
        "status": "validated_only" if dry_run else "sensor_forecast_only",
        "data_kind": "recorded_private", "sample_id": sample_id, "columns": columns,
        "models": {name: MODEL_IDS[name] for name in forecasters}, "model_load_s": load_s,
        "available_columns": list(table.columns), "source_rows": len(table), "csv_bytes": csv_bytes,
        "cadence": cadence_summary(table.timestamps_s), "clock_basis": "CSV_first_row_elapsed_s",
        "cross_stream_alignment_verified": False, "runs": runs,
    }))


if __name__ == "__main__":
    main()
