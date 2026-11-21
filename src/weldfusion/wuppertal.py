"""Wuppertal "Metal Arc Welding" dataset (Hahn et al., Zenodo 15497262, CC BY 4.0).

Each CSV row is one current pulse: 200 voltage (V_000..V_199) and 200 current
(I_000..I_199) samples, a per-pulse ``label`` (1 good, 0 bad, -1 unlabeled),
``welding_run_id`` and ``exp_id``. Rows are in time order *within* a contiguous
file block of a run; a run is stored as up to four blocks (``segments``), which
look like separate passes and are never joined here.

The pulse period is not documented. If the 200 points are raw 100 kHz samples,
one pulse lasts 2 ms; that value is an *assumption* recorded in every export.
"""
from __future__ import annotations

import hashlib

import numpy as np

DATASET = "Metal Arc Welding (Hahn et al., University of Wuppertal)"
DOI = "10.5281/zenodo.15497262"
LICENCE = "CC-BY-4.0"
ASSUMED_PULSE_PERIOD_S = 0.002   # 200 samples at 100 kHz; not stated by the source
INTEL_SENSOR_STEP_S = 0.116      # median Intel pilot sensor gap

FEATURE_NAMES = (
    "i_mean", "i_std", "i_peak", "i_background", "peak_index", "pulse_width", "decay_index",
    "v_mean", "v_std", "v_peak", "v_background", "v_min_after_peak", "short_circuit_samples",
    "power_mean",
)


def pulse_features(V, I):
    """Hand-made per-pulse features; V and I are (n, 200) arrays."""
    V = np.asarray(V, dtype=np.float32)
    I = np.asarray(I, dtype=np.float32)
    n, width = I.shape
    peak_index = I.argmax(axis=1)
    i_peak = I.max(axis=1)
    i_bg = I[:, -50:].mean(axis=1)
    half = (i_peak + I.min(axis=1)) / 2
    after = np.arange(width)[None, :] > peak_index[:, None]
    # First sample after the peak within 20% of the way back to background.
    settle = after & (I < (i_bg + 0.2 * (i_peak - i_bg))[:, None])
    decay = np.where(settle.any(axis=1), settle.argmax(axis=1), width)
    v_after = np.where(after, V, np.inf)
    return np.column_stack([
        I.mean(axis=1), I.std(axis=1), i_peak, i_bg, peak_index, (I > half[:, None]).sum(axis=1), decay,
        V.mean(axis=1), V.std(axis=1), V.max(axis=1), V[:, -50:].mean(axis=1),
        np.where(np.isfinite(v_after.min(axis=1)), v_after.min(axis=1), V[:, -1]),
        (V < 10).sum(axis=1), (V * I).mean(axis=1),
    ]).astype(np.float32)


def segment_ids(run_ids):
    """Label each row '<run>/<block>' where blocks are contiguous file stretches of one run."""
    run_ids = np.asarray(run_ids).astype(str)
    out = np.empty(run_ids.size, dtype=object)
    for run in dict.fromkeys(run_ids):
        rows = np.flatnonzero(run_ids == run)
        block = np.concatenate([[0], np.cumsum(np.diff(rows) != 1)])
        out[rows] = [f"{run}/{b}" for b in block]
    return out


def label_stretches(labels):
    """[(start, stop, label)] runs of equal consecutive labels."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, labels.size]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(starts, stops)]


def segment_episode(run, exp, segment, V, I, labels, pulse_period_s=ASSUMED_PULSE_PERIOD_S,
                    window_s=INTEL_SENSOR_STEP_S):
    """One segment, averaged to Intel's ~9 Hz sensor timescale, as a WeldEpisode v1 dict.

    Telemetry rows average ``window_s / pulse_period_s`` consecutive pulses.
    Label stretches become events; times rely on the assumed pulse period.
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    labels = np.asarray(labels).astype(int)
    per = max(1, int(round(window_s / pulse_period_s)))
    n = (len(labels) // per) * per
    if n == 0:
        raise ValueError("segment shorter than one averaging window")
    feats = pulse_features(V[:n], I[:n])
    grouped = lambda x: x.reshape(-1, per, *x.shape[1:])
    vi = grouped(V[:n] * I[:n]).mean(axis=(1, 2))
    lab = grouped(labels[:n])
    telemetry = []
    for k in range(n // per):
        rows = slice(k * per, (k + 1) * per)
        labelled = lab[k][lab[k] >= 0]
        telemetry.append({
            "time_s": round(k * per * pulse_period_s, 6),
            "current_a": float(I[rows].mean()), "voltage_v": float(V[rows].mean()),
            "power_w": float(vi[k]),
            "current_peak_mean_a": float(feats[rows, FEATURE_NAMES.index("i_peak")].mean()),
            "current_background_mean_a": float(feats[rows, FEATURE_NAMES.index("i_background")].mean()),
            "short_circuit_pulses": int((feats[rows, FEATURE_NAMES.index("short_circuit_samples")] > 0).sum()),
            "pulses": per,
            "labelled_fraction": float(labelled.size / per),
            "good_fraction_of_labelled": float((labelled == 1).mean()) if labelled.size else None,
        })
    events = [{"time_s": round(a * pulse_period_s, 6), "end_s": round(b * pulse_period_s, 6),
               "type": "quality_label", "label": {1: "good", 0: "bad"}[lab_value], "pulses": b - a}
              for a, b, lab_value in label_stretches(labels[:n]) if lab_value >= 0]
    good = int((labels[:n] == 1).sum())
    bad = int((labels[:n] == 0).sum())
    identity = hashlib.sha256(f"{DOI}|{run}|{segment}".encode()).hexdigest()[:10]
    return {
        "schema_version": "1.0.0",
        "episode_id": f"wuppertal-run{run}-seg{segment}-{identity}",
        "source": {"kind": "recorded", "dataset": DATASET, "doi": DOI, "licence": LICENCE,
                   "attribution": "Hahn, Königsfeld, Tercan, Buchholz, Purrio, Angerhausen, "
                                  "Meyes, Meisen (2025). Metal Arc Welding. Zenodo.",
                   "welding_run_id": str(run), "exp_id": str(exp), "segment": int(segment),
                   "use": "open"},
        "process": {"process": "metal arc welding (pulsed waveform observed; process not stated)"},
        "recipe": {},
        "labels": {"granularity": "per_pulse", "good_pulses": good, "bad_pulses": bad,
                   "unlabelled_pulses": int(n - good - bad)},
        "timeline": {"time_basis": "pulse_index_times_assumed_period",
                     "clock_alignment_verified": False,
                     "pulse_period_s": pulse_period_s, "pulse_period_assumed": True,
                     "telemetry_window_s": per * pulse_period_s, "pulses_per_window": per,
                     "source_sample_rate_hz": 100_000},
        "streams": {"sensors": f"zenodo:{DOI}/welding_data.csv#welding_run_id={run}&segment={segment}"},
        "telemetry": telemetry, "events": events, "reviews": [],
    }
