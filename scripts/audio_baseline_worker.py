"""Modal GPU worker: download every FLAC by signed URL, build features, train, score.

Input on stdin (JSON): {"cohort": [...], "urls": {flac_key: signed_url}, "config": {...}}.
Output: exactly one JSON line on stdout holding reports for each split protocol.
Media stays in the container's temporary disk and memory; nothing is uploaded.
"""
import json
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from weldfusion import audio_experiment as ax

MAX_FLAC_BYTES = 20 * 1024 * 1024


def log(message):
    print(message, file=sys.stderr, flush=True)


def fetch(url, path):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as src, open(path, "wb") as dst:
                size = 0
                while block := src.read(1 << 20):
                    size += len(block)
                    if size > MAX_FLAC_BYTES:
                        raise RuntimeError("FLAC exceeds size limit")
                    dst.write(block)
            return
        except RuntimeError:
            raise
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("download failed after retries")  # never echo the URL


def main():
    job = json.load(sys.stdin)
    cohort, urls, config = job["cohort"], job["urls"], job["config"]
    n_fft = ax.matched_n_fft(config.get("paper_n_fft", 16384))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"{len(cohort)} welds, n_fft {n_fft}, device {device}")
    features, durations = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        def one(row):
            path = Path(tmp) / (row["sample_id"].replace("/", "__") + ".flac")
            fetch(urls[row["flac"]], path)
            data, rate = sf.read(str(path), dtype="float32", always_2d=True)
            path.unlink()
            if rate != ax.RELEASED_RATE_HZ or data.shape[1] != 1:
                raise RuntimeError(f"{row['sample_id']}: unexpected rate/channels")
            return row["sample_id"], ax.stft_log_magnitude(data[:, 0], n_fft).astype(np.float16), len(data) / rate
        with ThreadPoolExecutor(32) as pool:
            for i, (sid, feats, seconds) in enumerate(pool.map(one, cohort), 1):
                features[sid], durations[sid] = feats, seconds
                if i % 500 == 0:
                    log(f"features {i}/{len(cohort)}")
    lengths = np.array(list(durations.values()))
    reports = {"n_fft": n_fft, "device": device,
               "audio_seconds": {"min": float(lengths.min()), "median": float(np.median(lengths)),
                                 "max": float(lengths.max())},
               "runs": {}}
    variants = config.get("variants") or [{"name": "std_b256", "target": "standardized", "batch": 256, "lr": 1e-4}]
    for protocol in config["protocols"]:
        rows = ax.session_resplit(cohort, seed=config.get("split_seed", 7)) if protocol == "session" else cohort
        log(f"--- protocol {protocol}: {ax.split_audit(rows)['sessions_spanning_splits']} sessions span splits")
        for v in variants:
            started = time.time()
            log(f"--- variant {v['name']} ({protocol})")
            report = ax.run(rows, features, torch, device=device, bottleneck=config.get("bottleneck", 48),
                            epochs=config.get("epochs", 50), seed=config.get("seed", 7), log=log,
                            target=v["target"], batch=v["batch"], lr=v["lr"])
            report["paper_n_fft"] = config.get("paper_n_fft", 16384)
            report["train_seconds"] = round(time.time() - started)
            key = protocol if len(variants) == 1 else f"{protocol}/{v['name']}"
            reports["runs"][key] = report
            # Emit each finished run at once so a lost sandbox keeps earlier results.
            print(json.dumps({"partial_run": key, "report": report}), flush=True)
            log(f"{key}: validation AUC {report['validation']['overall_auc']:.4f} "
                f"test AUC {report['test']['overall_auc']:.4f} final train_mse "
                f"{report['train_mse_by_epoch'][-1]:.4f} ({report['train_seconds']} s)")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
