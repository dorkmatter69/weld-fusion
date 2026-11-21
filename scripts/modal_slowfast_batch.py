"""Stride-one SlowFast R101 features for Intel welds, in parallel Modal GPU Sandboxes.

    .venv/bin/modal run scripts/modal_slowfast_batch.py --limit 4             # benchmark + pilot equivalence
    .venv/bin/modal run scripts/modal_slowfast_batch.py --shards 16           # everything not yet exported

Reads AVIs from gs://<weldops-data>/staged/intel-welding/ (signer: read-only, that
prefix) and writes create-only float16 feature archives to
gs://<modal-weld-pilot>/features/slowfast-r101-v015/full/<session>/<weld>.npz.
Signed URLs travel on stdin only. Per-weld metadata goes to ~/weldops-private/intel/video/.
"""

import csv
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

import modal_slowfast_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "project-da693f7a-9b8c-40dd-a1d"
DATA_BUCKET = PROJECT + "-weldops-data"
FEATURE_BUCKET = PROJECT + "-modal-weld-pilot"
SOURCE_PREFIX = "staged/intel-welding/"
FEATURE_PREFIX = "features/slowfast-r101-v015/full/"
PILOT_REF = "features/slowfast-r101-v015/04-01-23-0024-00.npz"
PILOT_SUBDIR = "2_good_weld_2_02-09-23_Fe410/04-01-23-0024-00"
SIGNER = "weldops-modal-pilot@{}.iam.gserviceaccount.com".format(PROJECT)
MANIFEST = Path.home() / "weldops-private" / "intel" / "manifest.csv"
OUT = Path.home() / "weldops-private" / "intel" / "video"

app = modal.App("weldops-slowfast-batch-export")
image = pilot.image.add_local_file(str(ROOT / "scripts" / "slowfast_batch_worker.py"),
                                   "/opt/slowfast_batch_worker.py", copy=True)


def sign(uris, verb="GET", headers=None, duration="12h", batch=200):
    out = []
    for i in range(0, len(uris), batch):
        group = uris[i:i + batch]
        command = ["gcloud", "storage", "sign-url", *group, "--duration=" + duration,
                   "--project=" + PROJECT, "--region=us-west1",
                   "--impersonate-service-account=" + SIGNER, "--http-verb=" + verb, "--format=json"]
        if headers:
            command.append("--headers=" + headers)
        for attempt in range(5):
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode == 0:
                break
            reason = ([l for l in done.stderr.splitlines() if l.strip()] or ["no stderr"])[-1][:300]
            print("sign-url failed (attempt {}): {}".format(attempt + 1, reason), flush=True)
            time.sleep(10 * 2 ** attempt)
        else:
            raise RuntimeError("sign-url failed: " + reason)
        rows = json.loads(done.stdout)
        rows = rows if isinstance(rows, list) else [rows]
        if len(rows) != len(group):
            raise RuntimeError("sign-url count mismatch")
        for uri, row in zip(group, rows):
            url = row.get("signed_url")
            if not url or not url.startswith("https://") or not row.get("resource", uri).endswith(uri.split("/", 3)[-1]):
                raise RuntimeError("unexpected sign-url output")
            out.append(url)
    return out


def existing_outputs():
    done = subprocess.run(["gcloud", "storage", "ls", "-r",
                           "gs://{}/{}**".format(FEATURE_BUCKET, FEATURE_PREFIX)],
                          capture_output=True, text=True)
    if done.returncode != 0 and "matched no objects" not in done.stderr.lower() \
            and "one or more urls matched no objects" not in done.stderr.lower():
        raise RuntimeError("could not list existing feature objects")
    prefix = "gs://{}/{}".format(FEATURE_BUCKET, FEATURE_PREFIX)
    return {l[len(prefix):-4] for l in done.stdout.split() if l.startswith(prefix) and l.endswith(".npz")}


def run_shard(index, jobs, batch, gpu, log_path, lock):
    sandbox = modal.Sandbox.create("python", "/opt/slowfast_batch_worker.py", app=app, image=image,
                                   gpu=gpu, cpu=4, memory=16384, timeout=24 * 3600)
    ok = failed = 0
    try:
        payload = json.dumps({"jobs": jobs, "batch": batch}).encode()
        for i in range(0, len(payload), 256 * 1024):
            sandbox.stdin.write(payload[i:i + 256 * 1024])
            sandbox.stdin.drain()
        sandbox.stdin.write_eof()
        sandbox.stdin.drain()
        for line in sandbox.stdout:
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            record["shard"] = index
            with lock:
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            if "summary" in record:
                continue
            ok += record.get("ok", False)
            failed += not record.get("ok", False)
            if not record.get("ok") or (ok + failed) % 25 == 1 or "pilot_max_abs_diff" in record:
                print("[shard {}] {}/{} {}".format(index, ok + failed, len(jobs), json.dumps(
                    {k: record.get(k) for k in ("sample_id", "ok", "error", "frames", "fps", "gpu_s",
                                                "wall_s", "pilot_max_abs_diff", "pilot_max_abs_value")
                     if k in record})), flush=True)
        sandbox.wait()
        if sandbox.returncode != 0:
            tail = [l for l in sandbox.stderr.read().splitlines() if l.strip()][-3:]
            print("[shard {}] worker exit {}: {}".format(index, sandbox.returncode, " | ".join(tail)), flush=True)
    finally:
        sandbox.terminate()
    return ok, failed


@app.local_entrypoint()
def main(limit: int = 0, shards: int = 1, batch: int = 16, gpu: str = "A10G"):
    rows = list(csv.DictReader(MANIFEST.open()))
    subdirs = sorted({r["SUBDIRS"] for r in rows}, key=lambda s: (s != PILOT_SUBDIR, s))
    done = existing_outputs()
    todo = [s for s in subdirs if s not in done]
    if limit:
        todo = todo[:limit]
    print("{} welds total, {} already exported, {} to do in {} shard(s)".format(
        len(subdirs), len(done), len(todo), shards), flush=True)
    if not todo:
        return
    image.build(app)
    gets = sign(["gs://{}/{}{}/{}.avi".format(DATA_BUCKET, SOURCE_PREFIX, s, s.split("/")[-1]) for s in todo])
    puts = sign(["gs://{}/{}{}.npz".format(FEATURE_BUCKET, FEATURE_PREFIX, s) for s in todo], verb="PUT",
                headers="content-type=application/octet-stream,x-goog-if-generation-match=0")
    jobs = [{"sample_id": s, "get_url": g, "put_url": p} for s, g, p in zip(todo, gets, puts)]
    if todo[0] == PILOT_SUBDIR:
        jobs[0]["compare_pilot"] = sign(["gs://{}/{}".format(FEATURE_BUCKET, PILOT_REF)])[0]
    print("signed {} weld jobs".format(len(jobs)), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "slowfast-export-{}.jsonl".format(time.strftime("%Y%m%d-%H%M%S"))
    lock = threading.Lock()
    parts = [jobs[i::shards] for i in range(shards)]
    with ThreadPoolExecutor(shards) as pool:
        results = list(pool.map(lambda a: run_shard(a[0], a[1], batch, gpu, log_path, lock),
                                enumerate(parts)))
    print("done: {} ok, {} failed; log {}".format(sum(r[0] for r in results),
                                                   sum(r[1] for r in results), log_path), flush=True)
