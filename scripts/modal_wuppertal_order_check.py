"""Check whether Wuppertal rows are in time order, and plot good vs bad pulses.

Reads the CSV already in the `weldops-wuppertal-maw` Modal Volume (see
modal_wuppertal_profile.py). Writes JSON and PNGs to ~/weldops-private/wuppertal/.

    .venv/bin/modal run scripts/modal_wuppertal_order_check.py

Order tests, per run, in file order versus a shuffled copy of the same run:
- label switches between consecutive rows (time-ordered labels come in stretches);
- lag-1 autocorrelation of per-pulse features (slow process drift survives only in time order);
- end-to-start join of consecutive rows versus random pairs from the same run.
"""

import base64
import json
from pathlib import Path

import modal


VOLUME_PATH = "/data"
FILE_NAME = "welding_data.csv"
OUT = Path.home() / "weldops-private" / "wuppertal"
SEED = 0

app = modal.App("weldops-wuppertal-order-check")
volume = modal.Volume.from_name("weldops-wuppertal-maw")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy>=1.26,<3", "pandas>=2.2", "matplotlib>=3.8")


def lag1(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    if x.size < 3 or np.std(x) == 0:
        return None
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def summary(values):
    import numpy as np
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return None
    return {"n": int(values.size), "min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max())}


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=4, memory=32768, timeout=3600)
def check():
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    path = Path(VOLUME_PATH) / FILE_NAME
    header = list(pd.read_csv(path, nrows=0).columns)
    v_cols = [c for c in header if c.startswith("V_")]
    i_cols = [c for c in header if c.startswith("I_")]
    dtypes = {c: np.float32 for c in v_cols + i_cols}
    data = pd.read_csv(path, dtype=dtypes)
    V = data[v_cols].to_numpy()
    I = data[i_cols].to_numpy()
    labels = data["label"].to_numpy()
    runs = data["welding_run_id"].to_numpy()
    exps = data["exp_id"].to_numpy()
    rows = np.arange(len(data))

    features = {
        "mean_current": I.mean(axis=1), "peak_current": I.max(axis=1),
        "mean_voltage": V.mean(axis=1),
        "peak_index": I.argmax(axis=1).astype(float),
        # Samples above half the pulse's own peak: a width proxy.
        "pulse_width": (I > (I.min(axis=1, keepdims=True) + I.max(axis=1, keepdims=True)) / 2).sum(axis=1).astype(float),
    }

    rng = np.random.default_rng(SEED)
    per_run = []
    for run in pd.unique(runs):
        idx = rows[runs == run]          # file order
        shuffled = rng.permutation(idx)
        lab = labels[idx]
        lab_s = labels[shuffled]
        blocks = int(1 + np.sum(np.diff(idx) != 1))
        entry = {"run": str(run), "exp": str(exps[idx[0]]), "rows": int(idx.size), "file_blocks": blocks,
                 "label_switches": int(np.sum(lab[1:] != lab[:-1])),
                 "label_switches_shuffled": int(np.sum(lab_s[1:] != lab_s[:-1])),
                 "longest_label_stretch": int(max(len(s) for s in np.split(lab, np.flatnonzero(np.diff(lab)) + 1))),
                 "lag1_file": {}, "lag1_shuffled": {}}
        for name, values in features.items():
            entry["lag1_file"][name] = lag1(values[idx])
            entry["lag1_shuffled"][name] = lag1(values[shuffled])
        # Join: end of row r-1 to start of row r, file order vs random pairs.
        join_file = np.abs(I[idx[1:], 0] - I[idx[:-1], -1])
        a, b = rng.integers(0, idx.size, (2, idx.size - 1))
        join_random = np.abs(I[idx[b], 0] - I[idx[a], -1])
        entry["median_join_current_file"] = float(np.median(join_file))
        entry["median_join_current_random"] = float(np.median(join_random))
        per_run.append(entry)

    def ratio(key_file, key_shuffled):
        return summary([r[key_file] / r[key_shuffled] if r[key_shuffled] else None for r in per_run])

    overall = {
        "runs": len(per_run),
        "label_switch_ratio_file_over_shuffled": ratio("label_switches", "label_switches_shuffled"),
        "longest_label_stretch": summary([r["longest_label_stretch"] for r in per_run]),
        "file_blocks_per_run": summary([r["file_blocks"] for r in per_run]),
        "lag1_file": {n: summary([r["lag1_file"][n] for r in per_run]) for n in features},
        "lag1_shuffled": {n: summary([r["lag1_shuffled"][n] for r in per_run]) for n in features},
        "join_current_file_over_random": ratio("median_join_current_file", "median_join_current_random"),
    }

    # Waveforms: mean and 10-90% band per experiment and label, plus feature distributions.
    waveforms, images = {}, {}
    exp_ids = sorted(pd.unique(exps), key=str)
    fig, axes = plt.subplots(2, len(exp_ids), figsize=(4.2 * len(exp_ids), 6.4), sharex=True, squeeze=False)
    colors = {0: "#c0392b", 1: "#2471a3", -1: "#909497"}
    names = {0: "bad", 1: "good", -1: "unlabeled"}
    for col, exp in enumerate(exp_ids):
        waveforms[str(exp)] = {}
        for label in (1, 0, -1):
            mask = (exps == exp) & (labels == label)
            if not mask.any():
                continue
            waveforms[str(exp)][names[label]] = {"n": int(mask.sum())}
            for row, (signal, unit) in enumerate(((I, "A"), (V, "V"))):
                block = signal[mask]
                mean = block.mean(axis=0)
                lo, hi = np.percentile(block, [10, 90], axis=0)
                waveforms[str(exp)][names[label]]["current" if row == 0 else "voltage"] = {
                    "mean": mean.round(3).tolist()}
                ax = axes[row][col]
                if label != -1:
                    ax.fill_between(np.arange(mean.size), lo, hi, color=colors[label], alpha=0.15, linewidth=0)
                ax.plot(mean, color=colors[label], linewidth=1.6 if label != -1 else 1,
                        linestyle="-" if label != -1 else ":",
                        label="{} (n={:,})".format(names[label], int(mask.sum())))
                ax.set_ylabel("{} [{}]".format("Current" if row == 0 else "Voltage", unit))
        axes[0][col].set_title("Experiment {}".format(exp))
        axes[0][col].legend(fontsize=8, frameon=False)
        axes[1][col].set_xlabel("Sample index within row (0-199)")
    fig.suptitle("Wuppertal metal arc welding: mean pulse per label (band = 10-90th percentile)")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    images["pulses_by_label.png"] = base64.b64encode(buffer.getvalue()).decode()

    # One run's per-pulse features in file order, to see drift and label stretches.
    longest = max(per_run, key=lambda r: r["rows"])
    idx = rows[runs.astype(str) == longest["run"]]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, name in zip(axes, ("peak_current", "mean_voltage", "pulse_width")):
        ax.plot(features[name][idx], linewidth=0.4, color="#34495e")
        ax.set_ylabel(name.replace("_", " "))
    lab = labels[idx]
    for label in (0, 1):
        axes[0].scatter(np.flatnonzero(lab == label), np.full((lab == label).sum(), features["peak_current"][idx].max()),
                        s=2, color=colors[label], label=names[label])
    axes[0].legend(fontsize=8, frameon=False, markerscale=4)
    axes[-1].set_xlabel("Row position within run (file order)")
    fig.suptitle("Run {} (experiment {}): pulse features in file order; dots = labels".format(
        longest["run"], longest["exp"]))
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    images["run_in_file_order.png"] = base64.b64encode(buffer.getvalue()).decode()

    feature_by_label = {name: {names[l]: summary(values[labels == l]) for l in (1, 0, -1)}
                        for name, values in features.items()}
    return {"overall": overall, "per_run": per_run, "feature_by_label": feature_by_label,
            "waveforms": waveforms, "images": images}


@app.local_entrypoint()
def main():
    result = check.remote()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, encoded in result.pop("images").items():
        (OUT / name).write_bytes(base64.b64decode(encoded))
    (OUT / "order_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overall": result["overall"], "feature_by_label": result["feature_by_label"]}, indent=2))
    print("Wrote {}".format(OUT))
