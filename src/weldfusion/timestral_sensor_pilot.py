"""Bounded, sensor-only Timestral forecast of one locally authorized CSV.

This checks forecast quality on the last rows of one weld. It does not assign
defect labels, align cameras or audio, or estimate anomaly-detection AUC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import load_sensor_table

MODEL_ID = "mistralai/Timestral-5M"


def prepare(table, columns, horizon=32, context=256):
    """Reject missing, irregular or ambiguous observations before forecasting."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("select distinct numeric sensor columns")
    absent = [name for name in columns if name not in table.columns]
    if absent:
        raise ValueError(f"sensor columns unavailable: {absent}; available: {table.columns}")
    if table.timestamps_s is None:
        raise ValueError("CSV needs recorded Date and Time columns")
    if not 1 <= horizon <= 1024 or not 64 <= context <= 32768:
        raise ValueError("horizon must be 1..1024 rows and context 64..32768 rows")
    if len(table) < context + horizon:
        raise ValueError(f"need at least {context + horizon} rows; found {len(table)}")
    times = np.asarray(table.timestamps_s, dtype=float)
    gaps = np.diff(times)
    if not np.all(np.isfinite(gaps)) or np.any(gaps <= 0):
        raise ValueError("sensor timestamps must be finite and strictly increasing")
    median_gap = float(np.median(gaps))
    if float(np.max(np.abs(gaps - median_gap)) / median_gap) > 0.25:
        raise ValueError("irregular sensor cadence; review gaps and resample before Timestral")
    selected = table.values[:, [table.columns.index(name) for name in columns]]
    if not np.all(np.isfinite(selected)):
        raise ValueError("selected sensor channels contain non-finite values")
    # Only rows before the held-out horizon are visible to the model.
    history = selected[-horizon-context:-horizon].T.astype(np.float32)
    observed = selected[-horizon:].T.astype(np.float32)
    return history, observed, {"cadence_hz": 1 / median_gap,
                               "context_rows": context, "forecast_rows": horizon,
                               "forecast_start_elapsed_s": float(times[-horizon]),
                               "forecast_end_elapsed_s": float(times[-1])}


def score_forecast(model, history, observed, columns):
    """One causal forecast; metrics use held-out rows only after inference."""
    forecast = model.forecast(history[None, :, :], prediction_length=observed.shape[1])
    median = np.asarray(forecast.median)
    low = np.asarray(forecast.quantile(0.05))
    high = np.asarray(forecast.quantile(0.95))
    expected = (1, len(columns), observed.shape[1])
    if any(array.shape != expected or not np.all(np.isfinite(array)) for array in (median, low, high)):
        raise ValueError("Timestral returned invalid forecast shape or values")
    if np.any(low > high):
        raise ValueError("Timestral quantile bounds are reversed")
    return {name: {"mae": float(np.mean(np.abs(observed[i] - median[0, i]))),
                   "fraction_outside_90pct_interval": float(np.mean(
                       (observed[i] < low[0, i]) | (observed[i] > high[0, i]))),
                   "mean_90pct_interval_width": float(np.mean(high[0, i] - low[0, i]))}
            for i, name in enumerate(columns)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="local authorized weld sensor CSV")
    parser.add_argument("output", type=Path, help="local JSON report; keep private")
    parser.add_argument("--columns", nargs="+", required=True, help="exact numeric CSV headers")
    parser.add_argument("--context", type=int, default=256, help="history rows (default: 256)")
    parser.add_argument("--horizon", type=int, default=32, help="held-out rows (default: 32)")
    parser.add_argument("--dry-run", action="store_true", help="validate CSV without loading weights")
    args = parser.parse_args(argv)
    table = load_sensor_table(args.csv)
    history, observed, details = prepare(table, args.columns, args.horizon, args.context)
    report = {"status": "validated_only" if args.dry_run else "sensor_forecast_only",
              "data_kind": "recorded_private", "sample_id": args.csv.stem,
              "model": None if args.dry_run else MODEL_ID,
              "columns": args.columns, "source_rows": len(table), "clock_basis": "CSV_first_row_elapsed_s",
              "cross_stream_alignment_verified": False, **details}
    if not args.dry_run:
        try:
            from timestral import TimestralForecaster
        except ImportError as exc:
            raise SystemExit("Install PyTorch and the official mistralai/timestral package first") from exc
        model = TimestralForecaster.from_pretrained(MODEL_ID)
        report["forecast_metrics"] = score_forecast(model, history, observed, args.columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {report['status']} report to {args.output}")


if __name__ == "__main__":
    main()
