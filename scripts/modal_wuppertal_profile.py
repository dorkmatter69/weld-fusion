"""Download the open Wuppertal "Metal Arc Welding" dataset into a Modal Volume and profile it.

Source: Hahn et al., Zenodo record 15497262 (DOI 10.5281/zenodo.15497262), CC BY 4.0.
Nothing is downloaded to this Mac. The profile JSON is written locally to
~/weldops-private/wuppertal/profile.json.

    .venv/bin/modal run scripts/modal_wuppertal_profile.py
"""

import json
from pathlib import Path

import modal


RECORD = "15497262"
FILE_NAME = "welding_data.csv"
URL = "https://zenodo.org/api/records/{}/files/{}/content".format(RECORD, FILE_NAME)
MD5 = "63489186e5ddae7fed6941444041cb68"
SIZE = 2_668_380_550
VOLUME_PATH = "/data"
REPORT = Path.home() / "weldops-private" / "wuppertal" / "profile.json"

app = modal.App("weldops-wuppertal-profile")
volume = modal.Volume.from_name("weldops-wuppertal-maw", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy>=1.26,<3", "pandas>=2.2")


def fetch(path):
    """Stream once from Zenodo; reuse the Volume copy when its size and MD5 match."""
    import hashlib
    import time
    import urllib.request

    def md5_of(p):
        digest = hashlib.md5()
        with open(p, "rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if path.exists() and path.stat().st_size == SIZE and md5_of(path) == MD5:
        return {"downloaded": False, "md5_verified": True, "bytes": SIZE}
    partial = path.with_suffix(".part")
    start = time.perf_counter()
    request = urllib.request.Request(URL, headers={"User-Agent": "weld-research-profiler"})
    digest, size = hashlib.md5(), 0
    with urllib.request.urlopen(request, timeout=120) as source, open(partial, "wb") as target:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            target.write(block)
    if size != SIZE or digest.hexdigest() != MD5:
        partial.unlink()
        raise RuntimeError("download mismatch: {} bytes, md5 {}".format(size, digest.hexdigest()))
    partial.rename(path)
    return {"downloaded": True, "md5_verified": True, "bytes": size,
            "download_s": round(time.perf_counter() - start, 1)}


def describe(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return {"n": int(values.size), "min": float(values.min()), "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=4, memory=16384, timeout=3600)
def profile():
    import numpy as np
    import pandas as pd

    path = Path(VOLUME_PATH) / FILE_NAME
    source = fetch(path)
    volume.commit()

    header = list(pd.read_csv(path, nrows=0).columns)
    v_cols = [c for c in header if c.startswith("V_")]
    i_cols = [c for c in header if c.startswith("I_")]
    other = [c for c in header if c not in v_cols and c not in i_cols]
    label_col = next((c for c in ("labels", "label") if c in header), None)
    run_col = next((c for c in ("welding_run_id", "welding_run_ids") if c in header), None)
    exp_col = next((c for c in ("exp_ids", "exp_id") if c in header), None)

    rows = 0
    nan_count = 0
    label_counts = {}
    runs = {}            # run -> {"n", "first", "last", "labels", "exps"}
    exp_runs = {}
    v_min, v_max, i_min, i_max = np.inf, -np.inf, np.inf, -np.inf
    v_sum = np.zeros(len(v_cols))
    i_sum = np.zeros(len(i_cols))
    v_cycle_mean, i_cycle_mean = [], []
    jumps_v, jumps_i, steps_v, steps_i = [], [], [], []
    prev = None          # (run, last V, last I) of the previous row in file order

    for chunk in pd.read_csv(path, chunksize=50_000):
        V = chunk[v_cols].to_numpy(dtype=float)
        I = chunk[i_cols].to_numpy(dtype=float)
        nan_count += int(np.isnan(V).sum() + np.isnan(I).sum())
        v_min, v_max = min(v_min, np.nanmin(V)), max(v_max, np.nanmax(V))
        i_min, i_max = min(i_min, np.nanmin(I)), max(i_max, np.nanmax(I))
        v_sum += np.nansum(V, axis=0)
        i_sum += np.nansum(I, axis=0)
        v_cycle_mean.append(np.nanmean(V, axis=1))
        i_cycle_mean.append(np.nanmean(I, axis=1))
        if len(steps_v) < 5:  # within-cycle sample-to-sample step, from a few chunks
            steps_v.append(np.abs(np.diff(V[:2000], axis=1)).ravel())
            steps_i.append(np.abs(np.diff(I[:2000], axis=1)).ravel())

        labels = chunk[label_col].to_numpy() if label_col else np.full(len(chunk), None)
        run_ids = chunk[run_col].to_numpy() if run_col else np.full(len(chunk), None)
        exp_ids = chunk[exp_col].to_numpy() if exp_col else np.full(len(chunk), None)
        for k in range(len(chunk)):
            index = rows + k
            label, run, exp = labels[k], run_ids[k], exp_ids[k]
            label_counts[str(label)] = label_counts.get(str(label), 0) + 1
            entry = runs.setdefault(str(run), {"n": 0, "first": index, "last": index,
                                               "labels": set(), "exps": set()})
            entry["n"] += 1
            entry["last"] = index
            entry["labels"].add(str(label))
            entry["exps"].add(str(exp))
            exp_runs.setdefault(str(exp), set()).add(str(run))
            if prev is not None and prev[0] == str(run):
                jumps_v.append(abs(V[k, 0] - prev[1]))
                jumps_i.append(abs(I[k, 0] - prev[2]))
            prev = (str(run), V[k, -1], I[k, -1])
        rows += len(chunk)

    run_sizes = [r["n"] for r in runs.values()]
    contiguous = sum(1 for r in runs.values() if r["last"] - r["first"] + 1 == r["n"])
    mixed_label_runs = sum(1 for r in runs.values() if len(r["labels"]) > 1)
    run_label = {}
    for r in runs.values():
        key = "/".join(sorted(r["labels"]))
        run_label[key] = run_label.get(key, 0) + 1
    v_cycle_mean = np.concatenate(v_cycle_mean)
    i_cycle_mean = np.concatenate(i_cycle_mean)
    step_v = float(np.median(np.concatenate(steps_v)))
    step_i = float(np.median(np.concatenate(steps_i)))
    jump_v = float(np.median(jumps_v)) if jumps_v else None
    jump_i = float(np.median(jumps_i)) if jumps_i else None
    v_mean, i_mean = v_sum / rows, i_sum / rows
    picks = [0, 25, 50, 100, 150, 199]

    return {
        "source": {"record": RECORD, "doi": "10.5281/zenodo.15497262", "licence": "CC-BY-4.0", **source},
        "columns": {"n_total": len(header), "n_voltage": len(v_cols), "n_current": len(i_cols),
                    "non_signal": other, "label_column": label_col, "run_column": run_col,
                    "experiment_column": exp_col},
        "rows": rows, "nan_values": nan_count,
        "label_counts_rows": label_counts,
        "runs": {"n": len(runs), "rows_per_run": describe(run_sizes),
                 "contiguous_in_file": contiguous, "runs_with_mixed_labels": mixed_label_runs,
                 "label_sets_per_run": run_label},
        "experiments": {"n": len(exp_runs),
                        "runs_per_experiment": describe([len(v) for v in exp_runs.values()])},
        "signal_ranges": {"voltage": [float(v_min), float(v_max)], "current": [float(i_min), float(i_max)]},
        "cycle_mean": {"voltage": describe(v_cycle_mean), "current": describe(i_cycle_mean)},
        "mean_waveform_at_index": {"index": picks,
                                   "voltage": [float(v_mean[p]) for p in picks],
                                   "current": [float(i_mean[p]) for p in picks]},
        "continuity": {
            "note": "jump = |first sample of row r - last sample of row r-1| within a run; "
                    "step = |sample k+1 - sample k| inside a row. jump ~ step suggests rows are back-to-back.",
            "median_within_row_step": {"voltage": step_v, "current": step_i},
            "median_row_boundary_jump": {"voltage": jump_v, "current": jump_i},
            "jump_to_step_ratio": {"voltage": jump_v / step_v if jump_v is not None and step_v else None,
                                   "current": jump_i / step_i if jump_i is not None and step_i else None},
        },
    }


@app.local_entrypoint()
def main():
    report = profile.remote()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Wrote {}".format(REPORT))
