"""Gated modality-shift fusion on the synthetic weld benchmark (RAVEN/MAG-style).

Video is the main modality. Audio and sensor are not concatenated; each computes
a bounded shift of the video representation, scaled by a learned scalar gate:

    h = e_video + g_audio * tanh(W_a A + c_a) + g_sensor * tanh(W_s S + c_s)
    g_m = sigmoid(u_m . [e_video, X_m, R_m] + b_m)
    p(fault) = sigmoid(w . h + b)

R_m are reliability signals a real system can observe: window presence,
within-window noise, and the clock-fit residual on a held-out landmark.

Stress tests corrupt one auxiliary modality in every test episode. Training sees
noise, missing windows and *flagged* misalignment on a random modality in a copy
of each training episode; *hidden* misalignment (clock residual looks normal) is
never shown in training. Everything is synthetic: nothing here transfers to the
Intel or Wuppertal recordings.

    PYTHONPATH=src python -m weldfusion.shift_fusion report.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .anomaly_eval import auc
from .multimodal_benchmark import generate, predict, representation, train_logistic, windows_for

AUX = ("audio", "sensor")
NOISE_SD = 0.6          # white noise per raw sample
BURST_SD = 0.15         # plus a random offset per one-second window
MISALIGN_S = 1.5        # clock-map error; faults last 2 s
TRAIN_CORRUPTIONS = ("noisy", "missing", "misaligned_flagged")
CONDITIONS = (
    ("clean", None, None),
    ("audio_noisy", "audio", "noisy"),
    ("audio_missing", "audio", "missing"),
    ("audio_misaligned_flagged", "audio", "misaligned_flagged"),
    ("audio_misaligned_hidden", "audio", "misaligned_hidden"),
    ("sensor_noisy", "sensor", "noisy"),
    ("sensor_missing", "sensor", "missing"),
    ("sensor_misaligned_hidden", "sensor", "misaligned_hidden"),
)
SEEDS = (42, 43, 44, 45, 46)


def corrupt(episode, mode, kind, rng):
    """Copy of ``episode`` with one modality degraded."""
    e = copy.deepcopy(episode)
    stream, clock = e["streams"][mode], e["clock_models"][mode]
    t = clock["slope"] * stream["local_t"] + clock["intercept"]
    if kind == "noisy":
        second = np.clip(np.floor(t), 0, 19).astype(int)
        bursts = rng.normal(0, BURST_SD, 20)[second][:, None]
        stream["values"] = stream["values"] + rng.normal(0, NOISE_SD, stream["values"].shape) + bursts
    elif kind == "missing":
        dropped = rng.choice(20, 8, replace=False)
        keep = ~np.isin(np.floor(t).astype(int), dropped)
        stream["local_t"], stream["values"] = stream["local_t"][keep], stream["values"][keep]
    elif kind in ("misaligned_flagged", "misaligned_hidden"):
        clock["intercept"] += MISALIGN_S * float(rng.choice([-1.0, 1.0]))
        if kind == "misaligned_flagged":   # the held-out landmark exposes the bad map
            clock["held_out_residual_s"] = MISALIGN_S
    else:
        raise ValueError(f"unknown corruption {kind}")
    return e


def _bin(episode, mode):
    """Per-second mean, within-second spread and presence on the calibrated clock."""
    stream, clock = episode["streams"][mode], episode["clock_models"][mode]
    t = clock["slope"] * stream["local_t"] + clock["intercept"]
    values = stream["values"]
    mean = np.zeros((20, values.shape[1]))
    spread, present = np.zeros(20), np.zeros(20)
    for s in range(20):
        picked = (t >= s) & (t < s + 1)
        if picked.any():
            mean[s] = values[picked].mean(axis=0)
            present[s] = 1
            spread[s] = values[picked].std(axis=0).mean() if picked.sum() > 1 else 0.0
    return mean, spread, present


def _context(x):
    """[x_t, x_{t-1}, x_{t+1}] with edge repetition."""
    prev = np.vstack([x[:1], x[:-1]])
    nxt = np.vstack([x[1:], x[-1:]])
    return np.hstack([x, prev, nxt])


def features(episode):
    blocks = {}
    bins = {mode: _bin(episode, mode) for mode in ("video", "audio", "sensor")}
    blocks["e"] = _context(bins["video"][0])
    for key, mode in (("A", "audio"), ("S", "sensor")):
        mean, spread, present = bins[mode]
        resid = episode["clock_models"][mode]["held_out_residual_s"]
        blocks[key] = _context(mean)
        blocks["R" + key.lower()] = np.column_stack([
            present, np.log(spread + 1e-3), np.full(20, np.log10(resid + 1e-4))])
    blocks["x7"] = np.column_stack([bins["video"][0], bins["audio"][0], bins["sensor"][0],
                                    bins["video"][2], bins["audio"][2], bins["sensor"][2]])
    return blocks, windows_for(episode)[1]


def stack(episodes):
    parts = [features(e) for e in episodes]
    blocks = {k: np.vstack([p[0][k] for p in parts]) for k in parts[0][0]}
    return blocks, np.concatenate([p[1] for p in parts])


class Standardizer:
    def __init__(self, blocks):
        self.stats = {k: (v.mean(0), np.where(v.std(0) < 1e-8, 1, v.std(0))) for k, v in blocks.items()}

    def __call__(self, blocks):
        return {k: (v - self.stats[k][0]) / self.stats[k][1] for k, v in blocks.items()}


# ---- gated shift model (numpy, analytic gradients) ----

def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def init_params(dims, rng):
    d = dims["e"]
    p = {"w": rng.normal(0, 0.1, d), "b": np.zeros(1)}
    for m, k, r in (("a", dims["A"], dims["Ra"]), ("s", dims["S"], dims["Rs"])):
        p["W" + m] = rng.normal(0, 0.1, (d, k))
        p["c" + m] = np.zeros(d)
        p["u" + m] = rng.normal(0, 0.1, d + k + r)
        p["g" + m] = np.zeros(1)
    return p


def forward(p, B, use_gate=True):
    e = B["e"]
    h = e.copy()
    cache = {}
    for m, X, R in (("a", B["A"], B["Ra"]), ("s", B["S"], B["Rs"])):
        shift = np.tanh(X @ p["W" + m].T + p["c" + m])
        G = np.column_stack([e, X, R])
        gate = _sigmoid(G @ p["u" + m] + p["g" + m][0]) if use_gate else np.ones(len(e))
        h = h + gate[:, None] * shift
        cache[m] = (X, G, shift, gate)
    return _sigmoid(h @ p["w"] + p["b"][0]), h, cache


def loss_and_grads(p, B, y, l2=1e-3, use_gate=True):
    prob, h, cache = forward(p, B, use_gate)
    n = len(y)
    eps = 1e-9
    reg = sum(np.sum(p[k] ** 2) for k in p if k[0] in "Wuw")
    loss = -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)) + 0.5 * l2 * reg
    d = (prob - y) / n
    g = {"w": h.T @ d + l2 * p["w"], "b": np.array([d.sum()])}
    dh = d[:, None] * p["w"][None, :]
    for m in ("a", "s"):
        X, G, shift, gate = cache[m]
        dpre = dh * gate[:, None] * (1 - shift ** 2)
        g["W" + m] = dpre.T @ X + l2 * p["W" + m]
        g["c" + m] = dpre.sum(0)
        if use_gate:
            dz = (dh * shift).sum(1) * gate * (1 - gate)
            g["u" + m] = G.T @ dz + l2 * p["u" + m]
            g["g" + m] = np.array([dz.sum()])
        else:
            g["u" + m] = l2 * p["u" + m]       # unused by the model; only the L2 term
            g["g" + m] = np.zeros(1)
    return loss, g


def train_gated(B, y, seed, steps=1500, lr=0.02, use_gate=True):
    """Full-batch Adam."""
    dims = {k: B[k].shape[1] for k in ("e", "A", "Ra", "S", "Rs")}
    p = init_params(dims, np.random.default_rng(seed))
    m1 = {k: np.zeros_like(v) for k, v in p.items()}
    m2 = {k: np.zeros_like(v) for k, v in p.items()}
    for t in range(1, steps + 1):
        _, g = loss_and_grads(p, B, y, use_gate=use_gate)
        for k in p:
            m1[k] = 0.9 * m1[k] + 0.1 * g[k]
            m2[k] = 0.999 * m2[k] + 0.001 * g[k] ** 2
            p[k] -= lr * (m1[k] / (1 - 0.9 ** t)) / (np.sqrt(m2[k] / (1 - 0.999 ** t)) + 1e-8)
    return p


# ---- experiment ----

def run_seed(seed, steps=1500):
    episodes = generate(seed)
    rng = np.random.default_rng(seed + 7)
    train_eps = []
    for e in (e for e in episodes if e["split"] == "TRAIN"):
        train_eps.append(e)
        train_eps.append(corrupt(e, str(rng.choice(AUX)), str(rng.choice(TRAIN_CORRUPTIONS)), rng))
    raw_train, y_train = stack(train_eps)
    std = Standardizer(raw_train)
    B = std(raw_train)
    concat = lambda b: np.hstack([b["e"], b["A"], b["Ra"], b["S"], b["Rs"]])
    models = {
        "video_only": train_logistic(raw_train["e"], y_train),
        "continuous": train_logistic(representation(raw_train["x7"], "continuous"), y_train),
        "concat_with_reliability": train_logistic(concat(raw_train), y_train),
    }
    gated = train_gated(B, y_train, seed, steps)
    no_gate = train_gated(B, y_train, seed, steps, use_gate=False)

    test = [e for e in episodes if e["split"] == "TEST"]
    results, gates = {}, {}
    for index, (name, mode, kind) in enumerate(CONDITIONS):
        crng = np.random.default_rng(seed * 1000 + index)
        eps = [corrupt(e, mode, kind, crng) if mode else e for e in test]
        raw, y = stack(eps)
        Bt = std(raw)
        scores = {
            "video_only": predict(models["video_only"], raw["e"]),
            "continuous": predict(models["continuous"], representation(raw["x7"], "continuous")),
            "concat_with_reliability": predict(models["concat_with_reliability"], concat(raw)),
            "gated_shift_no_gate": forward(no_gate, Bt, use_gate=False)[0],
            "gated_shift": forward(gated, Bt)[0],
        }
        _, _, cache = forward(gated, Bt)
        results[name] = {k: auc(v.tolist(), y.tolist()) for k, v in scores.items()}
        gates[name] = {"audio": float(cache["a"][3].mean()), "sensor": float(cache["s"][3].mean())}
    return {"seed": seed, "auc": results, "mean_gate": gates}


def run(seeds=SEEDS, steps=1500):
    per_seed = [run_seed(s, steps) for s in seeds]
    names = [c[0] for c in CONDITIONS]
    models = list(per_seed[0]["auc"]["clean"])
    summary = {c: {m: {"mean": float(np.mean([r["auc"][c][m] for r in per_seed])),
                       "sd": float(np.std([r["auc"][c][m] for r in per_seed]))} for m in models}
               for c in names}
    gate_summary = {c: {k: float(np.mean([r["mean_gate"][c][k] for r in per_seed])) for k in AUX}
                    for c in names}
    return {"protocol": "synthetic_gated_shift_fusion_v1", "data_kind": "simulation",
            "main_modality": "video", "seeds": list(seeds), "train_steps": steps,
            "corruption": {"noise_sd": NOISE_SD, "burst_sd": BURST_SD, "misalign_s": MISALIGN_S,
                           "train_corruptions": list(TRAIN_CORRUPTIONS),
                           "unseen_in_training": ["misaligned_hidden"]},
            "test_auc": summary, "mean_gate": gate_summary, "per_seed": per_seed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args(argv)
    report = run(tuple(args.seeds), args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote gated shift-fusion report to {args.output}")


if __name__ == "__main__":
    main()
