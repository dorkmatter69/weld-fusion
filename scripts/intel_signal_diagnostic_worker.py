"""What does one Intel 9 Hz current/voltage reading represent? (Modal CPU Sandbox)

Averaged over ~0.1 s, a pulsed arc gives smooth, unimodal readings with high
lag-1 autocorrelation. Instantaneous samples of a pulsed arc land randomly on
peaks and background: a wide, often bimodal spread and low autocorrelation.
Only summary statistics and binned counts leave the Sandbox; no raw rows.
"""

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from weldfusion.dataset import load_sensor_table
from timestral_pilot_worker import download


CHANNELS = ("Primary Weld Current", "Secondary Weld Voltage", "Feed", "CO2 Weld Flow")
ARC_ON_AMPS = 20.0


def channel_stats(x):
    x = np.asarray(x, dtype=float)
    diffs = np.abs(np.diff(x))
    sd = float(np.std(x))
    counts, edges = np.histogram(x, bins=20)
    lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.size > 2 and sd > 0 else None
    return {"n": int(x.size), "mean": float(np.mean(x)), "sd": sd,
            "coefficient_of_variation": sd / abs(float(np.mean(x))) if np.mean(x) else None,
            "p05": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)), "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)), "lag1_autocorrelation": lag1,
            "median_abs_row_change": float(np.median(diffs)),
            "median_abs_change_over_sd": float(np.median(diffs) / sd) if sd else None,
            "histogram": {"edges": [round(float(e), 3) for e in edges], "counts": counts.tolist()}}


def main():
    sample_id = os.environ["PILOT_SAMPLE_ID"]
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "{}.csv".format(sample_id)
        download(os.environ.pop("PILOT_GET_CSV"), csv_path)
        table = load_sensor_table(csv_path)
    current = table.values[:, table.columns.index("Primary Weld Current")]
    on = current > ARC_ON_AMPS
    # Longest contiguous arc-on stretch, trimmed 1 s each end to skip ignition and crater fill.
    best, start = (0, 0), None
    for k, flag in enumerate(np.append(on, False)):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            best = max(best, (k - start, start))
            start = None
    length, first = best
    times = np.asarray(table.timestamps_s, dtype=float) if table.timestamps_s is not None else None
    trim = int(round(1.0 / float(np.median(np.diff(times))))) if times is not None else 9
    steady = slice(first + trim, first + length - trim)
    report = {"sample_id": sample_id, "rows": len(table), "arc_on_threshold_A": ARC_ON_AMPS,
              "arc_on_rows": int(on.sum()), "longest_arc_on_rows": int(length),
              "steady_rows_used": int(max(0, length - 2 * trim)), "trim_rows_each_end": trim,
              "arc_on_elapsed_s": [float(times[first]), float(times[first + length - 1])] if times is not None and length else None,
              "channels": {}}
    for name in CHANNELS:
        if name in table.columns:
            report["channels"][name] = channel_stats(table.values[steady, table.columns.index(name)])
    print(json.dumps(report))


if __name__ == "__main__":
    main()
