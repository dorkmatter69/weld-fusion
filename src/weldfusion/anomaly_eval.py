"""Whole-weld anomaly evaluation after audio/video models produce errors.

Inputs are per-weld scores, not synchronized frames. Training Good scores set
normalization; validation chooses fusion weight; test is used exactly once.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev


def audio_mean(errors: list[float]) -> float:
    if not errors or not all(math.isfinite(x) for x in errors):
        raise ValueError("audio errors must be nonempty and finite")
    return fmean(errors)


def video_peak_mean(errors: list[float], window_frames: int) -> float:
    """Peak full-window moving mean, using a fixed frame count.

    Edge handling and actual frame timestamps need an explicit protocol before
    claiming exact numerical replication of the paper.
    """
    if window_frames < 1 or len(errors) < window_frames:
        raise ValueError("video requires at least one full window")
    if not all(math.isfinite(x) for x in errors):
        raise ValueError("video errors must be finite")
    total = sum(errors[:window_frames])
    peak = total / window_frames
    for i in range(window_frames, len(errors)):
        total += errors[i] - errors[i - window_frames]
        peak = max(peak, total / window_frames)
    return peak


def auc(scores: list[float], labels: list[int]) -> float:
    """Tie-aware ROC AUC: probability a defect outranks a Good weld."""
    if len(scores) != len(labels) or any(v not in (0, 1) for v in labels):
        raise ValueError("scores and binary labels must have equal length")
    if not all(math.isfinite(x) for x in scores):
        raise ValueError("scores must be finite")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both Good and defective welds")
    pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += sum(label for _, label in pairs[i:j]) * ((i + 1 + j) / 2)
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _split(value: str) -> str:
    v = value.strip().upper()
    if v in ("VAL", "VALID", "VALIDATION"):
        return "validation"
    if v in ("TRAIN", "TEST"):
        return v.lower()
    raise ValueError(f"unknown split: {value!r}")


def evaluate(records: list[dict]) -> dict:
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate sample_id")
    groups = {name: [] for name in ("train", "validation", "test")}
    for record in records:
        for key in ("audio_score", "video_score"):
            if not math.isfinite(record[key]):
                raise ValueError(f"nonfinite {key}: {record['sample_id']}")
        groups[_split(record["split"])].append(record)
    train = groups["train"]
    if not train or any(r["category"].strip().lower() != "good" for r in train):
        raise ValueError("training scores must contain Good welds only")
    stats = {}
    for mode in ("audio", "video"):
        scores = [r[f"{mode}_score"] for r in train]
        mean, std = fmean(scores), pstdev(scores)
        if std == 0:
            raise ValueError(f"zero training variance for {mode}")
        stats[mode] = {"mean": mean, "std": std}

    def normalized(record, mode):
        return (record[f"{mode}_score"] - stats[mode]["mean"]) / stats[mode]["std"]

    def labels(rows):
        return [int(r["category"].strip().lower() != "good") for r in rows]

    validation = groups["validation"]
    if not validation or not groups["test"]:
        raise ValueError("validation and test partitions are required")
    weights = [i / 100 for i in range(101)]
    candidates = []
    for audio_weight in weights:
        scores = [audio_weight * normalized(r, "audio") +
                  (1 - audio_weight) * normalized(r, "video") for r in validation]
        candidates.append(auc(scores, labels(validation)))
    best_index = max(range(len(weights)), key=lambda i: candidates[i])
    best_weight = weights[best_index]

    def report(rows):
        scored = {
            mode: [normalized(r, mode) for r in rows] for mode in ("audio", "video")
        }
        scored["fusion"] = [best_weight * a + (1 - best_weight) * v
                            for a, v in zip(scored["audio"], scored["video"])]
        result = {"counts": dict(Counter(r["category"] for r in rows)),
                  "auc": {mode: auc(values, labels(rows)) for mode, values in scored.items()},
                  "per_defect_auc": {}}
        defects = sorted({r["category"] for r in rows if r["category"].lower() != "good"})
        for defect in defects:
            chosen = [i for i, r in enumerate(rows)
                      if r["category"].lower() == "good" or r["category"] == defect]
            result["per_defect_auc"][defect] = {
                mode: auc([values[i] for i in chosen], [labels(rows)[i] for i in chosen])
                for mode, values in scored.items()
            }
        return result

    return {"protocol": "whole_weld_late_fusion_v1", "normalization_train_good": stats,
            "selected_audio_weight": best_weight, "selected_video_weight": 1 - best_weight,
            "validation": report(validation), "test": report(groups["test"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args(argv)
    records = json.loads(args.scores_json.read_text())
    result = evaluate(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote validation/test AUC to {args.output_json}")


if __name__ == "__main__":
    main()
