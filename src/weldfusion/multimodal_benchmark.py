"""Reproducible, entirely synthetic multimodal weld fusion benchmark.

This supervised toy task is deliberately separate from the Intel paper's
unsupervised real-data reproduction. No robot or cloud service is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .anomaly_eval import auc

MODALITIES = ("video", "audio", "sensor")
WIDTHS = {"video": 1, "audio": 1, "sensor": 2}
SEED = 42


def fusion_gate(data_kind, clock_evidence):
    """Restrict time-aligned fusion to independently checked clock maps."""
    verified = bool(clock_evidence.get("independent_holdout_passed"))
    if data_kind == "simulation" and verified:
        return {"aligned_fusion_allowed": True, "reason": "simulated independent clock landmarks passed"}
    if data_kind == "recorded" and verified and clock_evidence.get("review_status") == "approved":
        return {"aligned_fusion_allowed": True, "reason": "recorded independent landmarks approved"}
    return {"aligned_fusion_allowed": False, "reason": "independent recorded clock review is incomplete"}


def modality_profiles(episodes):
    """Measured properties of generated sources, not inferred real-data properties."""
    descriptions = {
        "video": ("continuous brightness proxy", "temporal", "arc appearance"),
        "audio": ("continuous energy proxy", "temporal", "acoustic events"),
        "sensor": ("two continuous process channels", "temporal", "current and gas flow"),
    }
    result = {}
    for mode in MODALITIES:
        lengths = [len(e["streams"][mode]["local_t"]) for e in episodes]
        deltas = np.concatenate([np.diff(e["streams"][mode]["local_t"]) for e in episodes])
        residuals = [e["clock_models"][mode]["held_out_residual_s"] for e in episodes]
        coverage = [windows_for(e)[0][:, 4 + MODALITIES.index(mode)] for e in episodes]
        representation, structure, relevance = descriptions[mode]
        result[mode] = {
            "representation": representation, "structure": structure,
            "task_relevance_hypothesis": relevance,
            "elements_per_episode": lengths[0], "median_rate_hz": round(float(1 / np.median(deltas)), 3),
            "max_held_out_clock_residual_ms": round(float(max(residuals) * 1000), 3),
            "missing_window_fraction": round(float(1 - np.mean(coverage)), 5),
            "noise_and_uncertainty": "simulated sample noise and device-clock fit error",
        }
    return result


def fit_clock(anchors_global, anchors_local):
    """Map device time to reference time using independent, held-out landmarks."""
    reference = np.asarray(anchors_global, float)
    local = np.asarray(anchors_local, float)
    if len(reference) < 3 or len(reference) != len(local):
        raise ValueError("at least three paired calibration landmarks required")
    slope, intercept = np.polyfit(local[:-1], reference[:-1], 1)
    residual = float(abs(slope * local[-1] + intercept - reference[-1]))
    if not np.isfinite(slope) or slope <= 0 or residual > 0.12:
        raise ValueError("clock fit failed independent landmark check")
    return float(slope), float(intercept), residual


def generate(seed=SEED, sessions=12, per_session=8):
    rng = np.random.default_rng(seed)
    episodes = []
    for session in range(sessions):
        split = "TRAIN" if session < 8 else "VALIDATION" if session < 10 else "TEST"
        for trial in range(per_session):
            category = ("Good", "Overlap", "Gas loss", "Spatter")[(trial + session) % 4]
            fault_at = float(rng.uniform(8, 13))
            base = rng.normal(0, 0.07, 3)
            streams, clocks = {}, {}
            for mode, hz in (("video", 5), ("audio", 25), ("sensor", 10)):
                t = np.arange(0, 20, 1 / hz)
                arc = ((t >= 3) & (t < 17)).astype(float)
                fault = ((t >= fault_at) & (t < fault_at + 2)).astype(float)
                if mode == "video":
                    values = (0.2 + 0.6 * arc + base[0] + rng.normal(0, 0.09, len(t))
                              + (0.30 * fault if category == "Overlap" else 0)
                              + (0.12 * fault if category == "Spatter" else 0))[:, None]
                elif mode == "audio":
                    values = (0.12 + 0.65 * arc + base[1] + rng.normal(0, 0.10, len(t))
                              + (0.26 * fault if category == "Overlap" else 0)
                              + (0.45 * fault if category == "Spatter" else 0)
                              + (0.22 * fault if category == "Gas loss" else 0))[:, None]
                else:
                    current = 0.12 + 0.7 * arc + base[2] + rng.normal(0, 0.05, len(t))
                    gas = 0.2 + 0.65 * arc + rng.normal(0, 0.05, len(t))
                    if category == "Gas loss":
                        gas -= 0.48 * fault
                    if category == "Overlap":
                        current += 0.10 * fault
                    values = np.stack((current, gas), axis=1)
                offset = float(rng.uniform(-0.9, 0.9))
                rate = float(rng.uniform(0.995, 1.005))
                local = offset + rate * t
                reference = np.array([2, 6, 14, 18.0])
                landmarks = offset + rate * reference + rng.normal(0, 0.005, 4)
                slope, intercept, residual = fit_clock(reference, landmarks)
                streams[mode] = {"local_t": local, "values": values}
                clocks[mode] = {"slope": slope, "intercept": intercept,
                                "held_out_residual_s": residual,
                                "ground_truth_offset_s": offset, "ground_truth_rate": rate}
            episodes.append({"id": f"synthetic-s{session:02d}-w{trial:02d}",
                             "session_id": f"synthetic-s{session:02d}", "split": split,
                             "category": category, "fault_at": fault_at,
                             "streams": streams, "clock_models": clocks})
    return episodes


def windows_for(episode, aligned=True):
    """One-second feature intervals, missing-data masks and their true labels."""
    x = np.zeros((20, 7), float)  # video, audio, sensor[2], presence[3]
    for i, mode in enumerate(MODALITIES):
        stream = episode["streams"][mode]
        time = stream["local_t"]
        if aligned:
            clock = episode["clock_models"][mode]
            time = clock["slope"] * time + clock["intercept"]
        column = (0, 1, 2)[i]
        for second in range(20):
            picked = (time >= second) & (time < second + 1)
            if picked.any():
                x[second, column:column + WIDTHS[mode]] = stream["values"][picked].mean(axis=0)
                x[second, 4 + i] = 1
    label = np.zeros(20, dtype=int)
    if episode["category"] != "Good":
        label = np.array([int(second < episode["fault_at"] + 2 and second + 1 > episode["fault_at"])
                          for second in range(20)])
    return x, label


def train_logistic(x, y, seed=SEED, steps=400):
    """Small regularized logistic classifier with train-only normalization."""
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-8] = 1
    z = (x - mean) / std
    weight = np.zeros(z.shape[1])
    bias = 0.0
    for _ in range(steps):
        logits = np.clip(z @ weight + bias, -30, 30)
        pred = 1 / (1 + np.exp(-logits))
        grad = pred - y
        weight -= 0.08 * (z.T @ grad / len(y) + 0.002 * weight)
        bias -= 0.08 * float(grad.mean())
    return (mean, std, weight, bias)


def predict(model, x):
    mean, std, weight, bias = model
    return 1 / (1 + np.exp(-np.clip(((x - mean) / std) @ weight + bias, -30, 30)))


def representation(x, kind, edges=None):
    if kind == "late_video":
        return x[:, [0]]
    if kind == "late_audio":
        return x[:, [1]]
    if kind == "late_sensor":
        return x[:, [2, 3]]
    if kind == "continuous":
        return np.column_stack((x, x[:, 0] * x[:, 1], x[:, 1] * x[:, 2], x[:, 0] * x[:, 2]))
    if kind == "tokenized":
        if edges is None:
            raise ValueError("train-only token boundaries required")
        parts = [np.eye(4)[np.searchsorted(edges[i], x[:, i])] for i in range(4)]
        return np.column_stack(parts + [x[:, 4:]])
    raise ValueError("unknown representation")


def metrics(y, scores):
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    threshold = 0.5
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    return {"auc": auc(scores.tolist(), y.tolist()),
            "precision_at_0_5": tp / (tp + fp) if tp + fp else 0.0,
            "recall_at_0_5": tp / (tp + fn) if tp + fn else 0.0,
            "positive_windows": int(y.sum()), "negative_windows": int(len(y) - y.sum())}


def run(seed=SEED):
    episodes = generate(seed)
    clock_evidence = {"independent_holdout_passed": all(
        e["clock_models"][mode]["held_out_residual_s"] <= 0.12
        for e in episodes for mode in MODALITIES), "source": "synthetic_landmarks"}
    gate = fusion_gate("simulation", clock_evidence)
    if not gate["aligned_fusion_allowed"]:
        raise ValueError("synthetic clock calibration failed: aligned fusion disabled")
    aligned = {e["id"]: windows_for(e, True) for e in episodes}
    naive = {e["id"]: windows_for(e, False) for e in episodes}
    def collect(split, data):
        chosen = [e for e in episodes if e["split"] == split]
        return np.concatenate([data[e["id"]][0] for e in chosen]), np.concatenate([data[e["id"]][1] for e in chosen])
    train_x, train_y = collect("TRAIN", aligned)
    edges = [np.quantile(train_x[:, i], [0.25, 0.5, 0.75]) for i in range(4)]
    models = {kind: train_logistic(representation(train_x, kind, edges), train_y)
              for kind in ("late_video", "late_audio", "late_sensor", "continuous", "tokenized")}
    naive_x, _ = collect("TRAIN", naive)
    models["uncalibrated_continuous"] = train_logistic(representation(naive_x, "continuous"), train_y)
    val_x, val_y = collect("VALIDATION", aligned)
    val_scores = {kind: predict(model, representation(val_x, kind, edges))
                  for kind, model in models.items() if kind != "uncalibrated_continuous"}
    late_weights = [i / 20 for i in range(21)]
    best_weight = max(late_weights, key=lambda w: auc((w * val_scores["late_video"] +
                        (1 - w) * val_scores["late_audio"]).tolist(), val_y.tolist()))
    test_x, test_y = collect("TEST", aligned)
    test_naive, _ = collect("TEST", naive)
    scores = {kind: predict(model, representation(test_x, kind, edges))
              for kind, model in models.items() if kind != "uncalibrated_continuous"}
    scores["uncalibrated_continuous"] = predict(models["uncalibrated_continuous"],
                                              representation(test_naive, "continuous"))
    scores["late_fusion"] = best_weight * scores["late_video"] + (1 - best_weight) * scores["late_audio"]
    test_episodes = [e for e in episodes if e["split"] == "TEST"]
    records = []
    for n, e in enumerate(test_episodes):
        beginning = n * 20
        for second in range(20):
            records.append({"episode_id": e["id"], "session_id": e["session_id"],
                            "category": e["category"], "time_s": second + 0.5,
                            "label": int(test_y[beginning + second]),
                            "scores": {k: round(float(v[beginning + second]), 6) for k, v in scores.items()}})
    by_category = {}
    for category in ("Overlap", "Gas loss", "Spatter"):
        selected = [i for i, r in enumerate(records) if r["category"] in ("Good", category)]
        by_category[category] = {kind: metrics(test_y[selected], value[selected]) for kind, value in scores.items()}
    return {"protocol": "synthetic_multimodal_fusion_v1", "seed": seed,
            "data_kind": "simulation", "training": "supervised_synthetic_only",
            "clock_alignment": "affine_from_three_landmarks_fourth_held_out",
            "clock_evidence": clock_evidence, "fusion_gate": gate,
            "modality_profiles": modality_profiles(episodes),
            "session_count": len({e["session_id"] for e in episodes}),
            "splits": {s: sum(e["split"] == s for e in episodes) for s in ("TRAIN", "VALIDATION", "TEST")},
            "late_video_weight_selected_on_validation": best_weight,
            "results": {kind: metrics(test_y, value) for kind, value in scores.items()},
            "per_category": by_category, "test_windows": records}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote synthetic multimodal report to {args.output}")


if __name__ == "__main__":
    main()
