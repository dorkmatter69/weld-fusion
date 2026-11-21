"""Bounded, sensor-only forecast of one locally authorized weld CSV.

Compares pretrained forecasters (Timestral-5M, TimesFM 3.0) with naive
baselines on held-out rows of one weld. It does not assign defect labels, align
cameras or audio, or estimate anomaly-detection AUC.

Window: ``arc_on`` (default) holds out the last rows of the steady arc-on
segment; ``file_end`` holds out the file's last rows, which may be after arc-off.

Cadence: the models receive values with no timestamps, so they assume equal steps.
- ``strict``: use rows as-is only if every used gap is within 25% of the median.
- ``rows``: treat each row as one step regardless of its real gap.
- ``resample``: linearly interpolate the history onto an even grid (refused if any
  gap exceeds ``max_gap_steps`` grid steps), forecast on the grid, then interpolate
  the forecast back to the *real* held-out timestamps. Scoring always uses real
  held-out readings, never interpolated ones.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from .dataset import load_sensor_table

MODEL_IDS = {"timestral": "mistralai/Timestral-5M", "timesfm": "google/timesfm-3.0-pytorch"}
MODEL_ID = MODEL_IDS["timestral"]
CADENCES = ("strict", "rows", "resample")


@dataclass
class Window:
    history: np.ndarray                 # (V, context) model input, equal steps
    observed: np.ndarray                # (V, horizon) real held-out readings
    forecast_steps: int                 # steps the model must produce
    step_s: float | None                # grid step (resample) else None
    observed_offsets_s: np.ndarray | None  # held-out times after the last history point
    details: dict


def arc_on_segment(table, column="Primary Weld Current", threshold=20.0, trim_s=1.0):
    """Row slice of the longest run above ``threshold``, trimmed at both ends."""
    if column not in table.columns:
        raise ValueError(f"arc column unavailable: {column}")
    on = np.append(table.values[:, table.columns.index(column)] > threshold, False)
    best, start = (0, 0), None
    for k, flag in enumerate(on):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            best = max(best, (k - start, start))
            start = None
    length, first = best
    if length == 0:
        raise ValueError("no arc-on rows above threshold")
    if table.timestamps_s is not None:
        times = np.asarray(table.timestamps_s, dtype=float)
        a = int(np.searchsorted(times, times[first] + trim_s - 1e-9))
        b = int(np.searchsorted(times, times[first + length - 1] - trim_s + 1e-9, side="right"))
    else:
        a, b = first, first + length
    if b <= a:
        raise ValueError("arc-on segment shorter than the trim")
    return a, b


def prepare(table, columns, horizon=32, context=256, cadence="strict", window="file_end",
            arc_column="Primary Weld Current", arc_threshold=20.0, trim_s=1.0,
            max_gap_steps=2.0) -> Window:
    """Select history and held-out rows; reject missing or ambiguous observations."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("select distinct numeric sensor columns")
    absent = [name for name in columns if name not in table.columns]
    if absent:
        raise ValueError(f"sensor columns unavailable: {absent}; available: {table.columns}")
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}")
    if table.timestamps_s is None and cadence != "rows":
        raise ValueError("CSV needs recorded Date and Time columns")
    if not 1 <= horizon <= 1024 or not 32 <= context <= 32768:
        raise ValueError("horizon must be 1..1024 rows and context 32..32768 rows")
    a, b = (0, len(table)) if window == "file_end" else arc_on_segment(
        table, arc_column, arc_threshold, trim_s)
    values = table.values[:, [table.columns.index(name) for name in columns]].T.astype(np.float64)
    if not np.all(np.isfinite(values[:, a:b])):
        raise ValueError("selected sensor channels contain non-finite values")
    times = None if table.timestamps_s is None else np.asarray(table.timestamps_s, dtype=float)
    held = slice(b - horizon, b)
    observed = values[:, held].astype(np.float32)
    details = {"window": window, "cadence": cadence, "segment_rows": [a, b],
               "forecast_rows": horizon, "variates": "forecast independently"}
    if times is not None:
        gaps = np.diff(times[a:b])
        if gaps.size == 0 or np.any(~np.isfinite(gaps)) or np.any(gaps <= 0):
            raise ValueError("sensor timestamps must be finite and strictly increasing")
        details.update({"segment_elapsed_s": [float(times[a]), float(times[b - 1])],
                        "forecast_elapsed_s": [float(times[held][0]), float(times[held][-1])]})

    if cadence in ("strict", "rows"):
        if b - a < context + horizon:
            raise ValueError(f"need {context + horizon} rows in the {window} segment; found {b - a}")
        used = slice(b - horizon - context, b)
        if times is not None:
            gaps = np.diff(times[used])
            median_gap = float(np.median(gaps))
            worst = float(np.max(np.abs(gaps - median_gap)) / median_gap)
            details.update({"median_gap_s": median_gap, "max_gap_deviation": worst})
            if cadence == "strict" and worst > 0.25:
                raise ValueError("irregular sensor cadence; review gaps and resample before forecasting")
        history = values[:, b - horizon - context:b - horizon].astype(np.float32)
        details["context_rows"] = context
        return Window(history, observed, horizon, None, None, details)

    # resample: even grid ending at the last history row.
    seg_t = times[a:b]
    step = float(np.median(np.diff(seg_t)))
    worst_gap = float(np.max(np.diff(seg_t)) / step)
    details.update({"grid_step_s": step, "max_gap_in_steps": worst_gap})
    if worst_gap > max_gap_steps:
        raise ValueError(f"gap of {worst_gap:.2f} steps exceeds {max_gap_steps}; possible missing rows")
    t_end = times[b - horizon - 1]
    grid = t_end - step * np.arange(context - 1, -1, -1)
    if grid[0] < times[a]:
        raise ValueError(f"need {context} grid steps of history in the {window} segment; "
                         f"only {int((t_end - times[a]) / step) + 1} fit")
    hist_t = times[a:b - horizon]
    history = np.stack([np.interp(grid, hist_t, row[a:b - horizon]) for row in values]).astype(np.float32)
    offsets = times[held] - t_end
    steps = int(math.ceil(offsets[-1] / step - 1e-9))
    details.update({"context_steps": context, "forecast_grid_steps": steps})
    return Window(history, observed, steps, step, offsets, details)


def interval_levels(levels, target=0.1):
    """Checkpoint quantile levels nearest a central interval; models reject other levels."""
    low = min(levels, key=lambda q: abs(q - target))
    high = min(levels, key=lambda q: abs(q - (1 - target)))
    if not low < 0.5 < high:
        raise ValueError(f"checkpoint quantile levels cannot form an interval: {levels}")
    return low, high


def to_observed(series, window: Window):
    """Map (V, steps, ...) model output onto the real held-out rows."""
    series = np.asarray(series, dtype=float)
    if window.step_s is None:
        return series[:, :window.observed.shape[1]]
    grid = window.step_s * np.arange(1, series.shape[1] + 1)
    flat = series.reshape(series.shape[0], series.shape[1], -1)
    out = np.stack([np.stack([np.interp(window.observed_offsets_s, grid, flat[v, :, j])
                              for j in range(flat.shape[2])], axis=-1) for v in range(flat.shape[0])])
    return out.reshape((series.shape[0], window.observed.shape[1]) + series.shape[2:])


def score_quantiles(levels, quantiles, window: Window, columns, interval_target=0.1):
    """Metrics against real held-out rows; quantiles is (V, steps, Q)."""
    levels = tuple(float(q) for q in levels)
    quantiles = np.asarray(quantiles, dtype=float)
    expected = (len(columns), window.forecast_steps, len(levels))
    if quantiles.shape != expected or not np.all(np.isfinite(quantiles)):
        raise ValueError(f"forecaster returned shape {quantiles.shape}, expected {expected}")
    low_q, high_q = interval_levels(levels, interval_target)
    mid = min(levels, key=lambda q: abs(q - 0.5))
    q = to_observed(quantiles, window)
    median, low, high = (q[..., levels.index(level)] for level in (mid, low_q, high_q))
    if np.any(low > high):
        raise ValueError("quantile bounds are reversed")
    obs = window.observed
    channels = {name: {"mae": float(np.mean(np.abs(obs[i] - median[i]))),
                       "fraction_outside_interval": float(np.mean((obs[i] < low[i]) | (obs[i] > high[i]))),
                       "mean_interval_width": float(np.mean(high[i] - low[i]))}
                for i, name in enumerate(columns)}
    return {"interval_quantiles": [low_q, high_q], "median_quantile": mid, "channels": channels}


def baselines(window: Window, columns):
    """Point-forecast references: repeat the last value; repeat the context mean."""
    obs = window.observed
    refs = {"persistence": window.history[:, -1], "context_mean": window.history.mean(axis=1)}
    return {name: {"channels": {col: {"mae": float(np.mean(np.abs(obs[i] - ref[i])))}
                                for i, col in enumerate(columns)}}
            for name, ref in refs.items()}


def load_forecaster(name):
    """Return run(history (V, C), steps) -> (levels, quantiles (V, steps, Q))."""
    if name == "timestral":
        from timestral import TimestralForecaster
        model = TimestralForecaster.from_pretrained(MODEL_IDS[name])

        def run(history, steps):
            out = model.forecast(np.asarray(history)[:, None, :], prediction_length=steps)
            return tuple(out.quantile_levels), np.asarray(out.quantiles)[:, 0]
        return run
    if name == "timesfm":
        from timesfm3 import TimesFM3Forecaster
        model = TimesFM3Forecaster.from_pretrained(MODEL_IDS[name], device="cpu")
        levels = tuple(float(q) for q in model.config.quantiles)

        def run(history, steps):
            outs = list(model.predict_batch([np.asarray(h) for h in history], horizon=steps,
                                            return_quantiles=True))
            q = np.stack([np.asarray(o.quantiles)[:steps] for o in outs])
            if q.shape[-1] == len(levels) + 1:   # leading mean column in some releases
                q = q[..., 1:]
            return levels, q
        return run
    raise ValueError(f"unknown model {name}; choose from {sorted(MODEL_IDS)}")


def run_pilot(table, columns, forecasters, cadences=CADENCES, **window_kwargs):
    """Every cadence x model, plus baselines. A cadence that fails preflight reports why."""
    runs = []
    for cadence in cadences:
        entry = {"cadence": cadence}
        try:
            window = prepare(table, columns, cadence=cadence, **window_kwargs)
        except ValueError as exc:
            entry["rejected"] = str(exc)
            runs.append(entry)
            continue
        entry["details"] = window.details
        entry["results"] = baselines(window, columns)
        for name, run in forecasters.items():
            start = time.perf_counter()
            levels, quantiles = run(window.history, window.forecast_steps)
            result = score_quantiles(levels, quantiles, window, columns)
            result["inference_s"] = round(time.perf_counter() - start, 3)
            result["model"] = MODEL_IDS[name]
            entry["results"][name] = result
        runs.append(entry)
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="local authorized weld sensor CSV")
    parser.add_argument("output", type=Path, help="local JSON report; keep private")
    parser.add_argument("--columns", nargs="+", required=True, help="exact numeric CSV headers")
    parser.add_argument("--models", nargs="*", default=["timestral"], choices=sorted(MODEL_IDS))
    parser.add_argument("--cadence", nargs="+", default=list(CADENCES), choices=CADENCES)
    parser.add_argument("--window", default="arc_on", choices=("arc_on", "file_end"))
    parser.add_argument("--context", type=int, default=176, help="history steps (default: 176)")
    parser.add_argument("--horizon", type=int, default=32, help="held-out rows (default: 32)")
    parser.add_argument("--dry-run", action="store_true", help="validate CSV without loading weights")
    args = parser.parse_args(argv)
    table = load_sensor_table(args.csv)
    forecasters = {} if args.dry_run else {name: load_forecaster(name) for name in args.models}
    runs = run_pilot(table, args.columns, forecasters, args.cadence, horizon=args.horizon,
                     context=args.context, window=args.window)
    report = {"status": "validated_only" if args.dry_run else "sensor_forecast_only",
              "data_kind": "recorded_private", "sample_id": args.csv.stem,
              "columns": args.columns, "source_rows": len(table),
              "clock_basis": "CSV_first_row_elapsed_s", "cross_stream_alignment_verified": False,
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {report['status']} report to {args.output}")


if __name__ == "__main__":
    main()
