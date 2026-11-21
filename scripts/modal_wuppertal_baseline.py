"""Wuppertal good/bad pulse baseline, label-confound probe, and WeldEpisode export.

Reads the CSV in the `weldops-wuppertal-maw` Modal Volume. Writes to
~/weldops-private/wuppertal/: baseline.json, episodes/*.json and episodes/*.mcap.

    .venv/bin/modal run scripts/modal_wuppertal_baseline.py

Evaluation never mixes pulses of one run between train and test:
- leave-one-experiment-out (4 folds): generalization to a new experiment;
- 5-fold grouped by run: generalization to a new run of a seen experiment.
Score = P(bad). Pulse-level AUC, label-stretch AUC (mean score per stretch), and
within-segment AUC (only segments holding both labels: no between-setting shortcut).
A "segment-mean" model sees only its segment's average features; if it matches the
pulse model, labels mostly identify segments/settings rather than pulse quality.
"""

import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
VOLUME_PATH = "/data"
FILE_NAME = "welding_data.csv"
OUT = Path.home() / "weldops-private" / "wuppertal"
INTEL_STEADY = {"current_mean_a": 178.8, "current_sd_a": 5.38, "current_lag1": 0.511,
                "voltage_mean_v": 17.31, "voltage_sd_v": 0.358, "voltage_lag1": 0.667}

app = modal.App("weldops-wuppertal-baseline")
volume = modal.Volume.from_name("weldops-wuppertal-maw")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26,<3", "pandas>=2.2", "scikit-learn>=1.4")
    .env({"PYTHONPATH": "/opt"})
    .add_local_dir(str(ROOT / "src" / "weldfusion"), "/opt/weldfusion", copy=True,
                  ignore=["**/__pycache__/**"])
)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=8, memory=32768, timeout=3600)
def run():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from weldfusion.wuppertal import (FEATURE_NAMES, label_stretches, pulse_features,
                                      segment_episode, segment_ids)

    path = Path(VOLUME_PATH) / FILE_NAME
    header = list(pd.read_csv(path, nrows=0).columns)
    v_cols = [c for c in header if c.startswith("V_")]
    i_cols = [c for c in header if c.startswith("I_")]
    data = pd.read_csv(path, dtype={c: np.float32 for c in v_cols + i_cols})
    V, I = data[v_cols].to_numpy(), data[i_cols].to_numpy()
    labels = data["label"].to_numpy().astype(int)
    runs = data["welding_run_id"].astype(str).to_numpy()
    exps = data["exp_id"].astype(str).to_numpy()
    seg = segment_ids(runs)
    del data

    X_feat = pulse_features(V, I)
    X_full = np.hstack([X_feat, I[:, ::10], V[:, ::10]])
    # Segment-mean features: the "which segment/setting is this" shortcut probe.
    seg_codes, seg_index = np.unique(seg, return_inverse=True)
    sums = np.zeros((seg_codes.size, X_feat.shape[1]))
    np.add.at(sums, seg_index, X_feat)
    X_segmean = (sums / np.bincount(seg_index)[:, None])[seg_index].astype(np.float32)

    lab = labels >= 0
    y = (labels[lab] == 0).astype(int)          # 1 = bad (the defect we detect)
    groups = {"leave_one_experiment_out": exps[lab], "grouped_by_run_5fold": runs[lab]}
    models = {
        "logistic_features": (lambda: make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced")), X_feat[lab]),
        "boosted_features_waveform": (lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0), X_full[lab]),
        "segment_mean_shortcut_probe": (lambda: make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced")), X_segmean[lab]),
    }

    # Label stretches within segments, for stretch-level scoring.
    stretch = np.full(labels.size, -1)
    sid = 0
    for code in seg_codes:
        rows = np.flatnonzero(seg == code)
        for a, b, value in label_stretches(labels[rows]):
            if value >= 0:
                stretch[rows[a:b]] = sid
                sid += 1
    stretch_l, seg_l = stretch[lab], seg[lab]

    results = {}
    for scheme, g in groups.items():
        splitter = (np.unique(g) if scheme.startswith("leave") else
                    list(GroupKFold(n_splits=5).split(np.zeros(g.size), y, g)))
        folds = ([(np.flatnonzero(g != e), np.flatnonzero(g == e)) for e in splitter]
                 if scheme.startswith("leave") else splitter)
        results[scheme] = {}
        for name, (make, X) in models.items():
            scores = np.zeros(y.size)
            fold_auc = []
            for train, test in folds:
                model = make().fit(X[train], y[train])
                scores[test] = model.predict_proba(X[test])[:, 1]
                fold_auc.append({"test_groups": sorted(set(g[test].tolist()))[:12],
                                 "n_test": int(test.size), "bad_rate": float(y[test].mean()),
                                 "auc": auc(y[test], scores[test])})
            st_ids = np.unique(stretch_l)
            st_score = np.array([scores[stretch_l == s].mean() for s in st_ids])
            st_y = np.array([y[stretch_l == s][0] for s in st_ids])
            within = [auc(y[seg_l == s], scores[seg_l == s]) for s in np.unique(seg_l)]
            within = [w for w in within if w is not None]
            results[scheme][name] = {
                "pulse_auc": auc(y, scores),
                "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, scores > 0.5)),
                "stretch_auc": auc(st_y, st_score), "n_stretches": int(st_ids.size),
                "within_segment_auc_median": float(np.median(within)) if within else None,
                "within_segment_auc_n": len(within), "folds": fold_auc,
            }

    seg_stats = []
    for code in seg_codes:
        rows = seg == code
        l = labels[rows]
        seg_stats.append({"segment": code, "exp": exps[rows][0], "pulses": int(rows.sum()),
                          "good": int((l == 1).sum()), "bad": int((l == 0).sum())})
    labelled = [s for s in seg_stats if s["good"] + s["bad"] > 0]
    mixed = [s for s in labelled if s["good"] and s["bad"]]
    confound = {
        "segments": len(seg_stats), "segments_with_labels": len(labelled),
        "segments_with_both_labels": len(mixed),
        "labelled_pulses_in_single_label_segments": float(
            sum(s["good"] + s["bad"] for s in labelled if not (s["good"] and s["bad"]))
            / max(1, sum(s["good"] + s["bad"] for s in labelled))),
    }
    by_label = {name: {"good": float(np.median(X_feat[labels == 1, k])),
                       "bad": float(np.median(X_feat[labels == 0, k]))}
                for k, name in enumerate(FEATURE_NAMES)}

    episodes = []
    for code in seg_codes:
        rows = np.flatnonzero(seg == code)
        run_id, block = code.split("/")
        try:
            episodes.append(segment_episode(run_id, exps[rows[0]], int(block), V[rows], I[rows], labels[rows]))
        except ValueError:
            continue

    return {"n_pulses": int(labels.size), "n_labelled": int(lab.sum()), "bad_rate_labelled": float(y.mean()),
            "results": results, "confound": confound, "segments": seg_stats,
            "feature_medians_by_label": by_label}, episodes


@app.local_entrypoint()
def main():
    import sys
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from weldfusion.episode_mcap import write_episode_mcap

    report, episodes = run.remote()
    folder = OUT / "episodes"
    folder.mkdir(parents=True, exist_ok=True)
    cur, volt = [], []
    for episode in episodes:
        stem = folder / episode["episode_id"]
        stem.with_suffix(".json").write_text(json.dumps(episode) + "\n")
        write_episode_mcap(episode, stem.with_suffix(".mcap"))
        c = np.array([r["current_a"] for r in episode["telemetry"]])
        v = np.array([r["voltage_v"] for r in episode["telemetry"]])
        if c.size > 20:
            cur.append((c.mean(), c.std(), np.corrcoef(c[:-1], c[1:])[0, 1]))
            volt.append((v.mean(), v.std(), np.corrcoef(v[:-1], v[1:])[0, 1]))
    cur, volt = np.array(cur), np.array(volt)
    report["bridge_vs_intel"] = {
        "note": "Wuppertal segments averaged to 58 pulses (0.116 s if the 2 ms pulse period holds) "
                "vs the Intel pilot's steady arc-on readings.",
        "intel_pilot": INTEL_STEADY,
        "wuppertal_segments_median": {
            "current_mean_a": float(np.median(cur[:, 0])), "current_sd_a": float(np.median(cur[:, 1])),
            "current_lag1": float(np.nanmedian(cur[:, 2])),
            "voltage_mean_v": float(np.median(volt[:, 0])), "voltage_sd_v": float(np.median(volt[:, 1])),
            "voltage_lag1": float(np.nanmedian(volt[:, 2])), "segments": int(len(cur))},
    }
    report["episodes_written"] = len(episodes)
    (OUT / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in ("n_pulses", "n_labelled", "bad_rate_labelled", "confound",
                                      "bridge_vs_intel", "episodes_written")}
    summary["results"] = {s: {m: {k: v for k, v in r.items() if k != "folds"} for m, r in ms.items()}
                          for s, ms in report["results"].items()}
    summary["folds_loeo"] = {m: [(f["test_groups"], f["auc"]) for f in r["folds"]]
                             for m, r in report["results"]["leave_one_experiment_out"].items()}
    print(json.dumps(summary, indent=2))
    print("Wrote {}".format(OUT))
