"""Paper-style video autoencoder versus masked temporal transformer.

Research protocol only. No private media or cloud access is performed here.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .anomaly_eval import auc, video_peak_mean


def load_index(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("index must be a nonempty JSON array")
    ids = set()
    for row in rows:
        if not all(row.get(k) for k in ("sample_id", "split", "category", "feature_path")):
            raise ValueError("each index row needs sample_id, split, category, feature_path")
        if row["sample_id"] in ids:
            raise ValueError("duplicate sample_id")
        ids.add(row["sample_id"])
        row["split"] = row["split"].strip().upper()
        if row["split"] == "VAL":
            row["split"] = "VALIDATION"
        if row["split"] not in ("TRAIN", "VALIDATION", "TEST"):
            raise ValueError("unknown split")
        if row["split"] == "TRAIN" and row["category"].strip().lower() != "good":
            raise ValueError("only Good welds may be used for training")
        feature_path = Path(row["feature_path"])
        row["feature_path"] = str(feature_path if feature_path.is_absolute() else path.parent / feature_path)
    if not all(any(r["split"] == name for r in rows) for name in ("TRAIN", "VALIDATION", "TEST")):
        raise ValueError("TRAIN, VALIDATION and TEST all required")
    for split in ("VALIDATION", "TEST"):
        cats = [r["category"].strip().lower() == "good" for r in rows if r["split"] == split]
        if not any(cats) or all(cats):
            raise ValueError(f"{split} requires Good and defective welds")
    # Session provenance is mandatory: adjacent windows and same-session runs
    # must never cross train/validation/test boundaries.
    sessions = {}
    for row in rows:
        if not row.get("session_id"):
            raise ValueError("session_id required for leakage audit")
        prior = sessions.setdefault(row["session_id"], row["split"])
        if prior != row["split"]:
            raise ValueError("session spans splits")
    return rows


def load_features(row: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(row["feature_path"], allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
        centers = archive["window_center_seconds"].astype(np.float64)
        if str(archive["sample_id"].item()) != row["sample_id"].split("/")[-1]:
            raise ValueError("feature sample identity mismatch")
        starts = archive["window_start_frame"]
    if (features.ndim != 2 or features.shape[1] != 2304 or len(features) != len(centers)
            or not len(features) or not np.isfinite(features).all() or not np.isfinite(centers).all()
            or len(starts) != len(features) or np.any(np.diff(starts) != 1)
            or np.any(np.diff(centers) <= 0)):
        raise ValueError("invalid stride-one SlowFast feature archive")
    return features, centers


def make_models(torch, d_model=128, depth=2):
    nn = torch.nn

    class PaperVideoAE(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [2304, 512, 256, 128, 64, 64, 64, 128, 256, 512, 2304]
            layers = []
            for i, (a, b) in enumerate(zip(widths, widths[1:])):
                layers.append(nn.Linear(a, b))
                if i < len(widths) - 2:
                    layers.append(nn.ReLU())
                if i in (1, 2, 3, 8):
                    layers.append(nn.Dropout(0.5))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class MaskedTemporal(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.LayerNorm(2304), nn.Linear(2304, d_model))
            self.mask_token = nn.Parameter(torch.zeros(d_model))
            self.positions = nn.Embedding(64, d_model)
            block = nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=4 * d_model,
                                               dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(block, depth, enable_nested_tensor=False)
            self.output = nn.Linear(d_model, 2304)

        def forward(self, x, mask):
            b, t, _ = x.shape
            pos = self.positions(torch.arange(t, device=x.device))[None]
            h = torch.where(mask[..., None], self.mask_token, self.input(x)) + pos
            return self.output(self.encoder(h))

    return PaperVideoAE(), MaskedTemporal()


def chunks(features: np.ndarray, length=64):
    for start in range(0, len(features), length):
        yield start, features[start:start + length]


def temporal_contexts(features: np.ndarray):
    """64 disjoint phases: tokens in one context share no raw video frames."""
    for phase in range(min(64, len(features))):
        indices = np.arange(phase, len(features), 64)
        for offset in range(0, len(indices), 64):
            chosen = indices[offset:offset + 64]
            yield chosen, features[chosen]


def score_sequence(model, kind, features, torch, device):
    model.eval()
    errors = np.empty(len(features), np.float64)
    with torch.no_grad():
        if kind == "ae":
            for start in range(0, len(features), 512):
                x = torch.as_tensor(features[start:start + 512], device=device)
                err = (model(x) - x).square().mean(-1)
                errors[start:start + len(x)] = err.cpu().numpy()
        else:
            # Four deterministic complementary masks score each position once.
            # A token cannot reconstruct itself from an unmasked copy.
            for chosen, block in temporal_contexts(features):
                x = torch.as_tensor(block[None], device=device)
                for phase in range(4):
                    mask = (torch.arange(len(block), device=device) % 4 == phase)[None]
                    pred = model(x, mask)
                    err = (pred - x).square().mean(-1)[0]
                    indices = mask[0].cpu().numpy()
                    errors[chosen[indices]] = err[mask[0]].cpu().numpy()
    return errors


def peak_two_seconds(errors, centers):
    # Full windows only, derived from actual exported center spacing.
    dt = float(np.median(np.diff(centers))) if len(centers) > 1 else 1.0
    if dt <= 0:
        raise ValueError("non-increasing centers")
    frames = min(len(errors), max(1, round(2.0 / dt)))
    return video_peak_mean(errors.tolist(), frames)


def report(rows, scores):
    selected = [r for r in rows if r["split"] == "TEST"]
    labels = [int(r["category"].lower() != "good") for r in selected]
    good = [r for r in selected if r["category"].lower() == "good"]
    defects = sorted({r["category"] for r in selected if r["category"].lower() != "good"})
    return {
        "overall_auc": auc([scores[r["sample_id"]] for r in selected], labels),
        "per_defect_auc": {
            defect: auc([scores[r["sample_id"]] for r in good + [r for r in selected if r["category"] == defect]],
                        [0] * len(good) + [1] * sum(r["category"] == defect for r in selected))
            for defect in defects
        },
        "counts": {"good": len(good), **{d: sum(r["category"] == d for r in selected) for d in defects}},
    }


def paired_overlap_bootstrap(rows, ae_scores, temporal_scores, seed=7, draws=1000):
    """Exploratory paired weld-level CI; session-cluster CI belongs in final study."""
    chosen = [r for r in rows if r["split"] == "TEST" and r["category"].lower() in ("good", "overlap")]
    good = [r for r in chosen if r["category"].lower() == "good"]
    overlap = [r for r in chosen if r["category"].lower() == "overlap"]
    if not good or not overlap:
        return None
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sample = [good[i] for i in rng.integers(len(good), size=len(good))]
        sample += [overlap[i] for i in rng.integers(len(overlap), size=len(overlap))]
        labels = [0] * len(good) + [1] * len(overlap)
        deltas.append(auc([temporal_scores[r["sample_id"]] for r in sample], labels)
                      - auc([ae_scores[r["sample_id"]] for r in sample], labels))
    return {"delta_auc": float(np.mean(deltas)), "percentile_95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
            "unit": "weld", "draws": draws}


def run(index, output, epochs=20, seed=7, device="cpu"):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rows = load_index(index)
    models = dict(zip(("ae", "temporal"), make_models(torch)))
    results = {}
    for kind, model in models.items():
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.0005 if kind == "ae" else 0.0001)
        best_auc, best_state, best_epoch = -1, None, None
        for epoch in range(epochs):
            model.train()
            training = [r for r in rows if r["split"] == "TRAIN"]
            random.shuffle(training)
            for row in training:
                features, _ = load_features(row)
                if kind == "ae":
                    samples = list(chunks(features, 256))
                else:
                    samples = list(temporal_contexts(features))
                random.shuffle(samples)
                for _, block in samples:
                    x = torch.as_tensor(block, device=device)
                    if kind == "ae":
                        loss = (model(x) - x).square().mean()
                    else:
                        x = x[None]
                        mask = (torch.rand((1, len(block)), device=device) < 0.25)
                        if not mask.any():
                            mask[:, 0] = True
                        prediction = model(x, mask)
                        loss = (prediction[mask] - x[mask]).square().mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            val = [r for r in rows if r["split"] == "VALIDATION"]
            val_scores = {}
            for row in val:
                f, t = load_features(row)
                val_scores[row["sample_id"]] = peak_two_seconds(score_sequence(model, kind, f, torch, device), t)
            metric = auc([val_scores[r["sample_id"]] for r in val],
                         [int(r["category"].lower() != "good") for r in val])
            if metric > best_auc:
                best_auc, best_epoch = metric, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        all_scores = {}
        for row in rows:
            f, t = load_features(row)
            all_scores[row["sample_id"]] = peak_two_seconds(score_sequence(model, kind, f, torch, device), t)
        results[kind] = {"best_validation_auc": best_auc, "best_epoch": best_epoch,
                         "test": report(rows, all_scores),
                         "scores": [{"sample_id": r["sample_id"], "split": r["split"],
                                     "category": r["category"], "video_score": all_scores[r["sample_id"]]}
                                    for r in rows]}
    overlap = {k: next((value for defect, value in v["test"]["per_defect_auc"].items()
                        if defect.lower() == "overlap"), None) for k, v in results.items()}
    ae_scores = {r["sample_id"]: r["video_score"] for r in results["ae"]["scores"]}
    temporal_scores = {r["sample_id"]: r["video_score"] for r in results["temporal"]["scores"]}
    comparison = paired_overlap_bootstrap(rows, ae_scores, temporal_scores, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": "paper_video_vs_masked_temporal_v1",
                                  "seed": seed, "epochs_max": epochs, "index": str(index),
                                  "overlap_test_auc": overlap, "overlap_paired_bootstrap": comparison,
                                  "models": results}, indent=2) + "\n")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("index", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.epochs < 1:
        p.error("epochs must be positive")
    run(args.index, args.output, args.epochs, args.seed, args.device)


if __name__ == "__main__":
    main()
