"""Run the bounded SlowFast benchmark inside the legacy Python GPU Sandbox."""

import collections
import json
import os
import tempfile
import time
import urllib.request


CHECKPOINT = (
    "https://download.openmmlab.com/mmaction/recognition/slowfast/"
    "slowfast_r101_4x16x1_256e_kinetics400_rgb/"
    "slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth"
)
CONFIG = (
    "/opt/mmaction2/configs/recognition/slowfast/"
    "slowfast_r101_r50_4x16x1_256e_kinetics400_rgb.py"
)


def expected_windows(frame_count):
    return max(0, frame_count - 63)


def benchmark(signed_url, sample_id, max_windows):
    if not 1 <= max_windows <= 32:
        raise ValueError("max_windows must be between 1 and 32")
    download_start = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "pilot.avi")
        try:
            with urllib.request.urlopen(signed_url, timeout=60) as source, open(path, "wb") as target:
                total_bytes = 0
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > 100 * 1024 * 1024:
                        raise RuntimeError("pilot AVI exceeds 100 MiB limit")
                    target.write(chunk)
        except Exception:
            raise RuntimeError("pilot AVI download failed; check signed URL and expiry") from None
        download_s = time.monotonic() - download_start
        return benchmark_video(path, sample_id, max_windows, download_s)


def benchmark_video(path, sample_id, max_windows, download_s):
    import cv2
    import mmcv
    import numpy as np
    import torch
    from mmaction.apis import init_recognizer

    cfg = mmcv.Config.fromfile(CONFIG)
    cfg.test_cfg = dict(feature_extraction=True)
    load_start = time.monotonic()
    model = init_recognizer(
        cfg, "/opt/checkpoints/slowfast.pth", device="cuda:0", use_frames=True
    )
    model.eval()
    load_s = time.monotonic() - load_start

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("could not open pilot AVI")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames < 64 or fps <= 0:
        raise ValueError("video has too few frames or invalid frame rate")

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    frames = collections.deque(maxlen=64)
    decode_s = preprocess_s = inference_s = 0.0
    decoded_frames = 0
    completed = 0
    try:
        for _ in range(64):
            t = time.monotonic()
            ok, frame = cap.read()
            decode_s += time.monotonic() - t
            if not ok:
                raise RuntimeError("AVI ended before first 64-frame window")
            frames.append(frame)
            decoded_frames += 1

        for index in range(min(max_windows, expected_windows(total_frames))):
            t = time.monotonic()
            selected = []
            for frame in list(frames)[::2]:
                h, w = frame.shape[:2]
                new_w = round(w * 256 / min(h, w))
                new_h = round(h * 256 / min(h, w))
                rgb = cv2.cvtColor(cv2.resize(frame, (new_w, new_h)), cv2.COLOR_BGR2RGB)
                top = (new_h - 224) // 2
                left = (new_w - 224) // 2
                selected.append((rgb[top:top + 224, left:left + 224].astype(np.float32) - mean) / std)
            clip = torch.from_numpy(np.stack(selected).transpose(3, 0, 1, 2)).unsqueeze(0).cuda()
            preprocess_s += time.monotonic() - t

            torch.cuda.synchronize()
            t = time.monotonic()
            with torch.no_grad():
                result = model(imgs=clip.unsqueeze(1), return_loss=False)
            torch.cuda.synchronize()
            inference_s += time.monotonic() - t
            if result.shape != (1, 2304):
                raise RuntimeError("unexpected feature shape: {}".format(result.shape))
            completed += 1
            if index + 1 < max_windows:
                t = time.monotonic()
                ok, frame = cap.read()
                decode_s += time.monotonic() - t
                if not ok:
                    break
                frames.append(frame)
                decoded_frames += 1
    finally:
        cap.release()

    full_windows = expected_windows(total_frames)
    estimated_compute_s = (
        download_s + load_s + decode_s / decoded_frames * (63 + full_windows)
        + (preprocess_s + inference_s) / completed * full_windows
    )
    return {
        "sample_id": sample_id,
        "checkpoint": CHECKPOINT,
        "mmaction2_version": "0.15.0",
        "gpu": "A10",
        "frame_count": total_frames,
        "fps": fps,
        "full_stride_one_windows": full_windows,
        "benchmarked_windows": completed,
        "feature_dimension": 2304,
        "download_s": round(download_s, 3),
        "model_load_s": round(load_s, 3),
        "decode_s": round(decode_s, 3),
        "preprocess_s": round(preprocess_s, 3),
        "gpu_inference_s": round(inference_s, 3),
        "estimated_one_weld_compute_s": round(estimated_compute_s, 1),
        "estimated_one_weld_gpu_usd": round(estimated_compute_s * 0.000306, 3),
        "estimate_excludes": ["image build", "container startup", "CPU/memory", "GCS egress"],
    }


if __name__ == "__main__":
    url = os.environ.pop("PILOT_SIGNED_URL")
    sample = os.environ["PILOT_SAMPLE_ID"]
    limit = int(os.environ["PILOT_MAX_WINDOWS"])
    print(json.dumps(benchmark(url, sample, limit)), flush=True)
