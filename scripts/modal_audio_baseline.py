"""Paper-style audio autoencoder baseline on the full Intel dataset, in a Modal GPU Sandbox.

    .venv/bin/modal run scripts/modal_audio_baseline.py --per-cell 3 --epochs 2   # smoke test
    .venv/bin/modal run scripts/modal_audio_baseline.py                           # full run

Reads the manifest copy in ~/weldops-private/intel/manifest.csv, signs one GET
URL per FLAC under gs://<weldops-data>/staged/intel-welding/ (signer may read
that prefix only), and sends them on the Sandbox's stdin, never in env, args or
logs. Reports go to ~/weldops-private/intel/audio/. Nothing is uploaded.
"""

import csv
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "project-da693f7a-9b8c-40dd-a1d"
BUCKET = PROJECT + "-weldops-data"
PREFIX = "staged/intel-welding/"
SIGNER = "weldops-modal-pilot@{}.iam.gserviceaccount.com".format(PROJECT)
MANIFEST = Path.home() / "weldops-private" / "intel" / "manifest.csv"
OUT = Path.home() / "weldops-private" / "intel" / "audio"

app = modal.App("weldops-intel-audio-baseline")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install("numpy>=1.26,<3", "soundfile>=0.12", "torch>=2.4")
    .add_local_dir(str(ROOT / "src" / "weldfusion"), "/opt/weldfusion", copy=True,
                  ignore=["**/__pycache__/**"])
    .add_local_file(str(ROOT / "scripts" / "audio_baseline_worker.py"),
                    "/opt/audio_baseline_worker.py", copy=True)
)


def sign_all(keys, duration="3h", batch=250, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    groups = [keys[i:i + batch] for i in range(0, len(keys), batch)]
    urls = {}
    with ThreadPoolExecutor(workers) as pool:
        for part in pool.map(_sign_group, groups, [duration] * len(groups)):
            urls.update(part)
            print("signed {}/{}".format(len(urls), len(keys)), flush=True)
    return urls


def _sign_group(group, duration):
    urls = {}
    command = ["gcloud", "storage", "sign-url", *["gs://{}/{}{}".format(BUCKET, PREFIX, k) for k in group],
               "--duration={}".format(duration), "--project={}".format(PROJECT), "--region=us-west1",
               "--impersonate-service-account={}".format(SIGNER), "--format=json"]
    for attempt in range(5):
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode == 0:
            break
        reason = ([l for l in done.stderr.splitlines() if l.strip()] or ["no stderr"])[-1][:300]
        print("sign-url batch failed (attempt {}): {}".format(attempt + 1, reason), flush=True)
        time.sleep(10 * 2 ** attempt)
    else:
        raise RuntimeError("sign-url failed after retries: " + reason)
    rows = json.loads(done.stdout)
    rows = rows if isinstance(rows, list) else [rows]
    if len(rows) != len(group):
        raise RuntimeError("gcloud returned {} URLs for {} objects".format(len(rows), len(group)))
    for key, row in zip(group, rows):
        url = row.get("signed_url") or row.get("signedUrl") or row.get("url")
        resource = row.get("resource") or row.get("url_resource") or ""
        if not url or not url.startswith("https://") or (resource and not resource.endswith(key)):
            raise RuntimeError("unexpected sign-url output")
        urls[key] = url
    return urls


@app.local_entrypoint()
def main(epochs: int = 50, bottleneck: int = 48, paper_n_fft: int = 16384,
         protocols: str = "paper,session", per_cell: int = 0, seed: int = 7, gpu: str = "A10G",
         variants: str = ""):
    """variants: comma list of target_bBATCH[_lrLR], e.g. "std_b256,raw_b256,raw_b32"."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from weldfusion.audio_experiment import cohort_from_manifest

    cohort = cohort_from_manifest(list(csv.DictReader(MANIFEST.open())))
    if per_cell:  # smoke test: first N welds of each (category, paper split), whole sessions not required
        cells = defaultdict(list)
        for row in cohort:
            cells[row["category"], row["split"]].append(row)
        cohort = [r for rows in cells.values() for r in rows[:per_cell]]
    print("{} welds".format(len(cohort)), flush=True)
    image.build(app)
    urls = sign_all([r["flac"] for r in cohort])
    parsed = []
    for name in [v.strip() for v in variants.split(",") if v.strip()]:
        parts = name.split("_")
        target = {"std": "standardized", "raw": "raw"}[parts[0]]
        batch = int(next(p[1:] for p in parts if p.startswith("b")))
        lr = float(next((p[2:] for p in parts if p.startswith("lr")), "1e-4"))
        parsed.append({"name": name, "target": target, "batch": batch, "lr": lr})
    config = {"variants": parsed, "epochs": epochs, "bottleneck": bottleneck, "paper_n_fft": paper_n_fft, "seed": seed,
              "protocols": [p.strip() for p in protocols.split(",") if p.strip()]}
    sandbox = modal.Sandbox.create(
        "python", "/opt/audio_baseline_worker.py", app=app, image=image,
        gpu=gpu, cpu=8, memory=32768, timeout=4 * 3600, env={"PYTHONPATH": "/opt"})
    try:
        payload = json.dumps({"cohort": cohort, "urls": urls, "config": config}).encode()
        for i in range(0, len(payload), 256 * 1024):  # Modal caps the stdin buffer
            sandbox.stdin.write(payload[i:i + 256 * 1024])
            sandbox.stdin.drain()
        sandbox.stdin.write_eof()
        sandbox.stdin.drain()
        del payload
        del urls
        tail = []
        for line in sandbox.stderr:
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-20:]
                if "epoch" not in line or line.split("/")[0].endswith(("0", "1")):
                    print("  [worker] " + line[:300], flush=True)
        output = sandbox.stdout.read()
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for line in output.splitlines():
            if line.startswith('{"partial_run"'):
                part = json.loads(line)
                (OUT / "run-{}-{}.json".format(part["partial_run"].replace("/", "_"), stamp)).write_text(
                    json.dumps({**part, "config": config}, indent=2) + "\n")
        try:
            sandbox.wait()
        except Exception as error:   # e.g. SandboxTerminatedError after partial results
            raise RuntimeError("sandbox ended early ({}); finished runs saved as run-*-{}.json".format(
                type(error).__name__, stamp))
        if sandbox.returncode != 0:
            raise RuntimeError("audio worker failed (exit {}):\n{}".format(sandbox.returncode, "\n".join(tail)))
        records = [json.loads(l) for l in output.splitlines() if l.startswith("{") and '"partial_run"' not in l[:20]]
        if len(records) != 1:
            raise RuntimeError("expected one result record")
        result = records[0]
        result["config"] = {**config, "welds": len(cohort), "per_cell": per_cell, "gpu": gpu}
        OUT.mkdir(parents=True, exist_ok=True)
        name = "audio-ae-fft{}-bn{}-ep{}{}{}-{}.json".format(
            paper_n_fft, bottleneck, epochs, "-smoke{}".format(per_cell) if per_cell else "",
            "-sweep" if len(parsed) > 1 else "",
            time.strftime("%Y%m%d-%H%M%S"))
        (OUT / name).write_text(json.dumps(result, indent=2) + "\n")
        for protocol, run in result["runs"].items():
            print("{:24s} val AUC {:.4f}  test AUC {:.4f}  ({} sessions span splits)".format(
                protocol, run["validation"]["overall_auc"], run["test"]["overall_auc"],
                run["split_audit"]["sessions_spanning_splits"]))
        print("Wrote {}".format(OUT / name))
    finally:
        sandbox.terminate()
