"""Batched stride-one SlowFast R101 feature export for many welds (legacy Python 3.8 GPU image).

Same model, checkpoint and preprocessing as slowfast_pilot_worker.py (short side
256, centre crop 224, ImageNet mean/std, 32 of each 64 frames), but each frame is
decoded and cropped once and windows run in batches. Input on stdin (JSON):
{"jobs": [{"sample_id", "get_url", "put_url"}], "batch": 16, "compare_pilot": optional}.
Prints one JSON line per weld (metadata only) and a final summary line.
"""
import base64
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request

import cv2
import mmcv
import numpy as np
import torch
from mmaction.apis import init_recognizer

CHECKPOINT_NAME = "slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth"
CONFIG = ("/opt/mmaction2/configs/recognition/slowfast/"
          "slowfast_r101_r50_4x16x1_256e_kinetics400_rgb.py")
MAX_AVI_BYTES = 200 * 1024 * 1024
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def log(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def fetch(url, path):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as src, open(path, "wb") as dst:
                size = 0
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_AVI_BYTES:
                        raise RuntimeError("AVI exceeds size limit")
                    dst.write(block)
            return size
        except RuntimeError:
            raise
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("AVI download failed after retries")


def put(url, payload):
    request = urllib.request.Request(url, data=payload, method="PUT", headers={
        "Content-Type": "application/octet-stream", "x-goog-if-generation-match": "0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if response.status in (200, 201):
                    return
        except urllib.error.HTTPError as error:
            if error.code == 412:
                raise RuntimeError("feature object already exists")
        except Exception:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError("feature upload failed after retries")


def decode(path):
    """All frames resized (short side 256) and centre-cropped to 224, uint8 RGB."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("could not open AVI")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        new_w, new_h = round(w * 256 / min(h, w)), round(h * 256 / min(h, w))
        rgb = cv2.cvtColor(cv2.resize(frame, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        top, left = (new_h - 224) // 2, (new_w - 224) // 2
        frames.append(rgb[top:top + 224, left:left + 224])
    cap.release()
    if len(frames) < 64 or fps <= 0:
        raise RuntimeError("video has too few frames or invalid fps")
    return np.stack(frames), fps


def extract(model, frames, batch):
    n = len(frames)
    windows = n - 63
    x = torch.from_numpy(frames).cuda().float()                 # (n, 224, 224, 3)
    x = ((x - torch.from_numpy(MEAN).cuda()) / torch.from_numpy(STD).cuda()).permute(3, 0, 1, 2)
    offsets = torch.arange(0, 64, 2, device="cuda")              # 32 of 64 frames
    out = np.empty((windows, 2304), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, windows, batch):
            idx = torch.arange(start, min(start + batch, windows), device="cuda")
            clip = x[:, idx[:, None] + offsets[None]]            # (3, B, 32, 224, 224)
            clip = clip.permute(1, 0, 2, 3, 4).unsqueeze(1)      # (B, 1, 3, 32, 224, 224)
            result = model(imgs=clip.contiguous(), return_loss=False)
            result = np.asarray(result)
            if result.shape != (len(idx), 2304):
                raise RuntimeError("unexpected feature shape {}".format(result.shape))
            out[start:start + len(idx)] = result
    if not np.isfinite(out).all():
        raise RuntimeError("nonfinite features")
    return out


def main():
    job = json.load(sys.stdin)
    torch.backends.cudnn.benchmark = True
    cfg = mmcv.Config.fromfile(CONFIG)
    cfg.test_cfg = dict(feature_extraction=True)
    model = init_recognizer(cfg, "/opt/checkpoints/slowfast.pth", device="cuda:0", use_frames=True)
    model.eval()
    batch = int(job.get("batch", 16))
    totals = {"welds": 0, "windows": 0, "seconds": 0.0, "failed": 0}
    with tempfile.TemporaryDirectory() as tmp:
        for item in job["jobs"]:
            started = time.time()
            record = {"sample_id": item["sample_id"]}
            try:
                avi = os.path.join(tmp, "weld.avi")
                record["avi_bytes"] = fetch(item["get_url"], avi)
                frames, fps = decode(avi)
                os.unlink(avi)
                t = time.time()
                features = extract(model, frames, batch)
                record["gpu_s"] = round(time.time() - t, 2)
                starts = np.arange(len(features), dtype=np.int32)
                buffer = tempfile.SpooledTemporaryFile()
                np.savez_compressed(
                    buffer, features=features.astype(np.float16), window_start_frame=starts,
                    window_end_frame=starts + 63, window_center_seconds=(starts + 31.5) / fps,
                    fps=np.float64(fps), frame_count=np.int32(len(frames)),
                    sample_id=np.array(item["sample_id"]), checkpoint=np.array(CHECKPOINT_NAME),
                    mmaction2_version=np.array("0.15.0"), feature_dtype=np.array("float16"))
                buffer.seek(0)
                payload = buffer.read()
                if item.get("compare_pilot"):
                    # Equivalence with the earlier one-window-at-a-time export.
                    ref_path = os.path.join(tmp, "pilot_ref.npz")
                    fetch(item["compare_pilot"], ref_path)
                    ref = np.load(ref_path)["features"]
                    record["pilot_max_abs_diff"] = float(np.abs(ref - features).max())
                    record["pilot_max_abs_value"] = float(np.abs(ref).max())
                put(item["put_url"], payload)
                record.update({"frames": len(frames), "fps": round(fps, 3), "windows": len(features),
                               "bytes": len(payload), "md5_base64": base64.b64encode(
                                   hashlib.md5(payload).digest()).decode("ascii"), "ok": True})
                totals["welds"] += 1
                totals["windows"] += len(features)
            except Exception as error:   # messages never include URLs
                record.update({"ok": False, "error": str(error)[:200]})
                totals["failed"] += 1
            record["wall_s"] = round(time.time() - started, 2)
            totals["seconds"] += record["wall_s"]
            print(json.dumps(record), flush=True)
    print(json.dumps({"summary": totals}), flush=True)


if __name__ == "__main__":
    main()
