"""Paper-style audio autoencoder baseline for the Intel welding dataset.

Stemmer et al. 2024 (arXiv 2409.02290), Sec. 3.1 / 4.1 / Table 5:
1D CNN autoencoder over STFT frames, trained on Good TRAIN welds only with MSE,
Adam + one-cycle LR (peak 1e-4), 50 epochs; per-weld score = mean frame error;
validation chooses hyperparameters (FFT window, bottleneck) and nothing else.

Deviations that cannot be avoided or are not specified by the paper:
- the released FLAC is 16 kHz, the paper used 192 kHz. Windows are matched in
  *duration* (16,384 / 192 kHz = 85.3 ms -> 1,366 samples), not in bandwidth;
- log1p magnitude with a Hann window (the paper does not state either);
- batch size, padding and the exact PReLU placement are not specified.

Two split protocols:
- ``paper``: the manifest's SPLIT column (weld-level; most sessions span splits);
- ``session``: every session sits in exactly one split (Good sessions 70/15/15,
  each defect category's sessions halved between VALIDATION and TEST).

This module never touches cloud storage. Workers feed it decoded audio.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

from .anomaly_eval import auc

PAPER_RATE_HZ = 192_000
RELEASED_RATE_HZ = 16_000
PAPER_N_FFTS = (4096, 16384, 32768, 65536)   # Sec. 4.1 grid
PAPER_BOTTLENECKS = (16, 32, 48, 64)         # Sec. 4.1 grid; 48 best
PAPER_BEST = {"paper_n_fft": 16384, "bottleneck": 48}
PAPER_AUDIO_TEST_AUC = 0.8460                # Table 9
PAPER_AUDIO_TEST_PER_DEFECT = {              # Table 9, FFT 16384, b.n. 48
    "Excessive_Penetration": 0.7807, "Burnthrough": 0.7018, "Porosity": 0.9949,
    "Porosity_w_Excessive_Penetration": 0.9659, "Undercut": 0.8989, "Overlap": 0.9743,
    "Lack_of_Fusion": 0.8372, "Excessive_Convexity": 0.8842, "Spatter": 0.7796,
    "Warping": 0.7648, "Crater_Cracks": 0.9453,
}
CONTEXT_FRAMES = 11  # Eq. 2 buffer: hop * (10 + n_fft/hop) samples -> 11 frames at 50% hop
SPLITS = ("TRAIN", "VALIDATION", "TEST")


# ---------------------------------------------------------------- features

def matched_n_fft(paper_n_fft: int, rate: int = RELEASED_RATE_HZ) -> int:
    """Window length at ``rate`` with the same duration as the paper's window."""
    return int(round(paper_n_fft * rate / PAPER_RATE_HZ))


def stft_log_magnitude(mono: np.ndarray, n_fft: int, hop: int | None = None) -> np.ndarray:
    """(frames, n_fft//2+1) float32 log1p |rfft(hann * frame)|, 50% hop by default."""
    hop = hop or n_fft // 2
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim != 1 or len(mono) < n_fft:
        raise ValueError("need a mono signal at least one window long")
    starts = np.arange(0, len(mono) - n_fft + 1, hop)
    frames = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[starts]
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(n_fft).astype(np.float32), axis=1))
    out = np.log1p(magnitude).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite audio features")
    return out


# ---------------------------------------------------------------- cohorts

def _split_name(value: str) -> str:
    value = value.strip().upper()
    return "VALIDATION" if value == "VAL" else value


def cohort_from_manifest(rows: list[dict]) -> list[dict]:
    """One record per weld with the paper's split; keys used by every later step."""
    cohort, seen = [], set()
    for row in rows:
        sample_id = row["SUBDIRS"].strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"missing or duplicate SUBDIRS: {sample_id!r}")
        seen.add(sample_id)
        stem = sample_id.split("/")[-1]
        split = _split_name(row["SPLIT"])
        if split not in SPLITS:
            raise ValueError(f"unknown split {row['SPLIT']!r}")
        cohort.append({"sample_id": sample_id, "session_id": row["DIRECTORY"].strip(),
                       "category": row["CATEGORY"].strip(), "paper_split": split,
                       "split": split, "flac": f"{sample_id}/{stem}.flac"})
    return cohort


def session_resplit(cohort: list[dict], seed: int = 7,
                    good_fractions=(0.70, 0.15, 0.15)) -> list[dict]:
    """Session-disjoint split with the paper's proportions. Deterministic per seed."""
    sessions = defaultdict(list)
    for row in cohort:
        sessions[row["session_id"]].append(row)
    categories = {s: {r["category"] for r in rs} for s, rs in sessions.items()}
    if any(len(c) != 1 for c in categories.values()):
        raise ValueError("session_resplit assumes one category per session")
    rng = random.Random(seed)
    assignment = {}
    by_category = defaultdict(list)
    for s, cats in categories.items():
        by_category[next(iter(cats))].append(s)
    for category, names in sorted(by_category.items()):
        names = sorted(names)
        rng.shuffle(names)
        good = category.lower() == "good"
        targets = dict(zip(SPLITS, good_fractions)) if good else {"VALIDATION": 0.5, "TEST": 0.5}
        if len(names) < len(targets):
            raise ValueError(f"{category}: {len(names)} sessions cannot fill {len(targets)} splits")
        total = sum(len(sessions[s]) for s in names)
        filled = Counter()
        # Seed each split with one session, then greedily fill the most under-target split.
        for split, s in zip(targets, names):
            assignment[s] = split
            filled[split] += len(sessions[s])
        for s in names[len(targets):]:
            split = min(targets, key=lambda k: filled[k] / (targets[k] * total))
            assignment[s] = split
            filled[split] += len(sessions[s])
    return [{**row, "split": assignment[row["session_id"]]} for row in cohort]


def split_audit(cohort: list[dict]) -> dict:
    sessions = defaultdict(set)
    for row in cohort:
        sessions[row["session_id"]].add(row["split"])
    table = Counter((row["category"], row["split"]) for row in cohort)
    good_train_sessions = {r["session_id"] for r in cohort
                           if r["split"] == "TRAIN" and r["category"].lower() == "good"}
    test_good = [r for r in cohort if r["split"] == "TEST" and r["category"].lower() == "good"]
    bad_train = [r for r in cohort if r["split"] == "TRAIN" and r["category"].lower() != "good"]
    if bad_train:
        raise ValueError("defective welds in TRAIN")
    return {
        "welds": len(cohort), "sessions": len(sessions),
        "sessions_spanning_splits": sum(len(v) > 1 for v in sessions.values()),
        "test_good_welds_sharing_a_train_session": sum(r["session_id"] in good_train_sessions
                                                       for r in test_good),
        "test_good_welds": len(test_good),
        "counts": {c: {s: table[c, s] for s in SPLITS}
                   for c in sorted({r["category"] for r in cohort})},
    }


# ---------------------------------------------------------------- model

def make_audio_ae(torch, n_bins: int, bottleneck: int = 48, width: int = 1024,
                  target: str = "standardized"):
    """Table 5: BN, 5 Conv1d encoders, 5 ConvTranspose1d decoders, kernel 3, stride 1.

    Leaky-ReLU after every layer except the last hidden one (PReLU); linear output.
    Padding 1 keeps the time length ("largely preserves the time dimension").
    ``target``: "standardized" reconstructs per-bin standardized input (fixed
    Good-train statistics); "raw" feeds and reconstructs the raw log magnitude,
    with the BatchNorm layer as the only input normalization (closest to Table 5).
    """
    if target not in ("standardized", "raw"):
        raise ValueError("target must be 'standardized' or 'raw'")
    nn = torch.nn
    enc = [n_bins, width, width, width, width, bottleneck]
    dec = [bottleneck, width, width, width, width, n_bins]
    layers = [nn.BatchNorm1d(n_bins)]
    for a, b in zip(enc, enc[1:]):
        layers += [nn.Conv1d(a, b, 3, padding=1), nn.LeakyReLU()]
    for i, (a, b) in enumerate(zip(dec, dec[1:])):
        layers.append(nn.ConvTranspose1d(a, b, 3, padding=1))
        if i < len(dec) - 3:
            layers.append(nn.LeakyReLU())
        elif i == len(dec) - 3:
            layers.append(nn.PReLU())

    class AudioAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(*layers)
            self.register_buffer("mean", torch.zeros(n_bins))
            self.register_buffer("std", torch.ones(n_bins))

        def forward(self, x):            # x: (batch, frames, bins)
            # Reconstruct per-bin standardized input with *fixed* Good-train
            # statistics. Using the BatchNorm output as target would let the
            # model shrink BN's scale and make the loss trivially small.
            z = (x - self.mean) / self.std if target == "standardized" else x
            y = self.network(z.transpose(1, 2)).transpose(1, 2)
            return y, z

    return AudioAE()


def chunk_starts(n_frames: int, length: int = CONTEXT_FRAMES) -> list[int]:
    """Non-overlapping chunk starts covering every frame; last chunk aligned to the end."""
    if n_frames < length:
        raise ValueError(f"weld has {n_frames} frames, needs >= {length}")
    starts = list(range(0, n_frames - length + 1, length))
    if starts[-1] + length < n_frames:
        starts.append(n_frames - length)
    return starts


def frame_errors(model, features: np.ndarray, torch, device, length=CONTEXT_FRAMES,
                 batch=512) -> np.ndarray:
    """Per-frame MSE (over bins); every frame counted once."""
    starts = chunk_starts(len(features), length)
    errors = np.full(len(features), np.nan)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            group = starts[i:i + batch]
            x = torch.as_tensor(np.stack([features[s:s + length] for s in group]),
                                dtype=torch.float32, device=device)
            pred, target = model(x)
            err = (pred - target).square().mean(-1).cpu().numpy()
            for s, e in zip(group, err):
                fill = np.isnan(errors[s:s + length])
                errors[s:s + length][fill] = e[fill]
    if np.isnan(errors).any():
        raise RuntimeError("frame not scored")
    return errors


def train_ae(features: dict[str, np.ndarray], train_ids: list[str], torch, device,
             bottleneck=48, epochs=50, lr=1e-4, batch=256, seed=7, length=CONTEXT_FRAMES,
             target="standardized", log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_bins = next(iter(features.values())).shape[1]
    model = make_audio_ae(torch, n_bins, bottleneck, target=target)
    stacked = np.concatenate([features[sid] for sid in train_ids])
    model.mean.copy_(torch.as_tensor(stacked.mean(0), dtype=torch.float32))
    model.std.copy_(torch.as_tensor(stacked.std(0) + 1e-3, dtype=torch.float32))
    del stacked
    model = model.to(device)
    # One epoch = one pass over the training frames: non-overlapping chunks with
    # a fresh random offset per weld each epoch, then shuffled.
    def epoch_index():
        out = []
        for sid in train_ids:
            n = len(features[sid])
            offset = int(rng.integers(0, min(length, n - length + 1)))
            out += [(sid, s) for s in range(offset, n - length + 1, length)]
        return out

    per_epoch = sum((len(features[sid]) - length) // length + 1 for sid in train_ids)
    steps = epochs * (math.ceil(per_epoch / batch) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    history = []
    for epoch in range(epochs):
        model.train()
        index = epoch_index()
        order = rng.permutation(len(index))
        total = 0.0
        for i in range(0, len(order), batch):
            chunk = [index[j] for j in order[i:i + batch]]
            x = torch.as_tensor(np.stack([features[sid][s:s + length] for sid, s in chunk]),
                                dtype=torch.float32, device=device)
            pred, target = model(x)
            loss = (pred - target).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(chunk)
        history.append(total / len(index))
        if log:
            log(f"epoch {epoch + 1}/{epochs} train_mse {history[-1]:.5f}")
    return model, history


# ---------------------------------------------------------------- evaluation

def evaluate(cohort: list[dict], scores: dict[str, float], split: str) -> dict:
    rows = [r for r in cohort if r["split"] == split and r["sample_id"] in scores]
    good = [r for r in rows if r["category"].lower() == "good"]
    defects = sorted({r["category"] for r in rows} - {r["category"] for r in good})
    if not good or not defects:
        raise ValueError(f"{split} needs Good and defective welds")
    labels = [int(r["category"].lower() != "good") for r in rows]
    per = {}
    for d in defects:
        chosen = good + [r for r in rows if r["category"] == d]
        per[d] = auc([scores[r["sample_id"]] for r in chosen],
                     [int(r["category"] == d) for r in chosen])
    return {"overall_auc": auc([scores[r["sample_id"]] for r in rows], labels),
            "per_defect_auc": per,
            "counts": {"Good": len(good), **{d: sum(r["category"] == d for r in rows) for d in defects}}}


def run(cohort: list[dict], features: dict[str, np.ndarray], torch, device="cpu",
        bottleneck=48, epochs=50, seed=7, log=None, target="standardized", batch=256,
        lr=1e-4) -> dict:
    """Train on Good TRAIN welds; score every weld; report VALIDATION and TEST."""
    missing = [r["sample_id"] for r in cohort if r["sample_id"] not in features]
    if missing:
        raise ValueError(f"{len(missing)} welds have no features, e.g. {missing[0]}")
    train_ids = [r["sample_id"] for r in cohort
                 if r["split"] == "TRAIN" and r["category"].lower() == "good"]
    model, history = train_ae(features, train_ids, torch, device, bottleneck=bottleneck,
                              epochs=epochs, seed=seed, log=log, target=target,
                              batch=batch, lr=lr)
    scores, max_scores = {}, {}
    for row in cohort:
        e = frame_errors(model, features[row["sample_id"]], torch, device)
        scores[row["sample_id"]] = float(e.mean())   # paper: expected value
        max_scores[row["sample_id"]] = float(e.max())
    return {
        "protocol": "stemmer2024_audio_ae_16k_v1",
        "bottleneck": bottleneck, "epochs": epochs, "seed": seed,
        "target": target, "batch": batch, "peak_lr": lr,
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "train_welds": len(train_ids), "train_mse_by_epoch": history,
        "split_audit": split_audit(cohort),
        "validation": evaluate(cohort, scores, "VALIDATION"),
        "test": evaluate(cohort, scores, "TEST"),
        "test_max_aggregation": evaluate(cohort, max_scores, "TEST"),
        "scores": [{"sample_id": r["sample_id"], "session_id": r["session_id"],
                    "split": r["split"], "category": r["category"],
                    "audio_score": scores[r["sample_id"]]} for r in cohort],
    }


def dumps(report: dict) -> str:
    return json.dumps(report, indent=2) + "\n"
