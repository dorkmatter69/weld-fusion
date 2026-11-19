"""Fit arc boundaries and evaluate separate, manually matched landmarks.

Input JSON contains "fit" (exactly two landmarks) and optional "holdout".
Each landmark has an "id" and two [earliest_s, latest_s] intervals:
"video_s" and "stream_s". Run separately for audio and sensor time.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def review(data):
    fit = data["fit"]
    holdout = data.get("holdout", [])
    events = fit + holdout
    if len(fit) != 2 or len({event["id"] for event in events}) != len(events):
        raise ValueError("provide two fit events and distinct landmark IDs")
    for event in events:
        for field in ("video_s", "stream_s"):
            low, high = event[field]
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError(f"invalid interval for {event['id']}: {field}")
    x = [mean(event["video_s"]) for event in fit]
    y = [mean(event["stream_s"]) for event in fit]
    if any(fit[0][field][1] >= fit[1][field][0] for field in ("video_s", "stream_s")):
        raise ValueError("fit intervals must be ordered and disjoint on both clocks")
    if any(not (fit[0][field][1] < event[field][0] <= event[field][1] < fit[1][field][0])
           for event in holdout for field in ("video_s", "stream_s")):
        raise ValueError("holdout landmarks must lie between the fit intervals")
    scale = (y[1] - y[0]) / (x[1] - x[0])
    models = {}
    for name, rate, offset in (
        ("offset_only", 1.0, mean(b - a for a, b in zip(x, y))),
        ("affine", scale, y[0] - scale * x[0]),
    ):
        residuals = []
        for event in holdout:
            vl, vh = event["video_s"]
            sl, sh = event["stream_s"]
            low, high = sl - (rate * vh + offset), sh - (rate * vl + offset)
            residuals.append({
                "id": event["id"],
                "midpoint_residual_s": mean((low, high)),
                "residual_interval_s": [low, high],
            })
        models[name] = {"scale": rate, "offset_s": offset, "holdout": residuals}
    return {
        "status": "holdout_scored" if holdout else "awaiting_independent_landmarks",
        "clock_alignment_verified": False,
        "interval_note": "Residual intervals reflect holdout annotations at fixed fit coefficients.",
        "models": models,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("landmarks", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.landmarks.read_text())), indent=2))
