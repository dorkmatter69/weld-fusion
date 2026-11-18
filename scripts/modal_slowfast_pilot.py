"""Bounded, read-only one-weld SlowFast feature-throughput benchmark.

Run only after creating the `weldops-pilot-gcs` Modal Secret with the GCS
HMAC keys documented in docs/MODAL_VIDEO_PILOT.md. No media is uploaded to
Modal storage; the AVI is read from GCS inside the ephemeral container.
"""

import os
import time
from pathlib import Path

import modal


SAMPLE_ID = os.environ.get("WELDOPS_SAMPLE_ID", "unset-sample")
BUCKET = os.environ.get("WELDOPS_GCS_BUCKET", "unset-bucket")
PREFIX = os.environ.get("WELDOPS_GCS_PREFIX", "unset-prefix/")
CHECKPOINT = (
    "https://download.openmmlab.com/mmaction/recognition/slowfast/"
    "slowfast_r101_4x16x1_256e_kinetics400_rgb/"
    "slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth"
)
CONFIG = (
    "/opt/mmaction2/configs/recognition/slowfast/"
    "slowfast_r101_r50_4x16x1_256e_kinetics400_rgb.py"
)

# PyTorch/MMCV are deliberately old to match the paper's MMAction2 v0.15.0.
# Build-time package and checkpoint downloads may be substantial. Keep the
# image separate from the repository's modern Python 3.14 environment.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04", add_python="3.8"
    )
    .apt_install("git", "curl", "libgl1", "libglib2.0-0", "ca-certificates")
    .run_commands(
        "python -m pip install --upgrade 'pip<24'",
        "python -m pip install 'numpy==1.23.5' 'opencv-python-headless==4.5.5.64'",
        "python -m pip install 'torch==1.8.1+cu111' 'torchvision==0.9.1+cu111' "
        "-f https://download.pytorch.org/whl/torch_stable.html",
        "python -m pip install 'mmcv-full==1.4.0' "
        "-f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html",
        "git clone --depth 1 --branch v0.15.0 "
        "https://github.com/open-mmlab/mmaction2.git /opt/mmaction2",
        "python -m pip install --no-deps -e /opt/mmaction2",
        "mkdir -p /opt/checkpoints && curl -fL --retry 3 "
        f"'{CHECKPOINT}' -o /opt/checkpoints/slowfast.pth",
    )
)

app = modal.App("weldops-slowfast-one-weld-benchmark")
gcs_read_only = modal.CloudBucketMount(
    BUCKET,
    bucket_endpoint_url="https://storage.googleapis.com",
    key_prefix=PREFIX,
    secret=modal.Secret.from_name(
        "weldops-pilot-gcs",
        required_keys=["GOOGLE_ACCESS_KEY_ID", "GOOGLE_ACCESS_KEY_SECRET"],
    ),
    read_only=True,
)


def expected_windows(frame_count: int) -> int:
    """Number of full 64-frame windows at one-frame stride."""
    return max(0, frame_count - 63)


@app.function(
    image=image,
    gpu="A10",
    cpu=2,
    memory=8192,
    volumes={"/pilot": gcs_read_only},
    timeout=600,
    startup_timeout=900,
    retries=0,
    max_containers=1,
)
def benchmark(sample_id: str, max_windows: int = 16) -> dict:
    import collections

    import cv2
    import mmcv
    import numpy as np
    import torch
    from mmaction.apis import init_recognizer

    if not 1 <= max_windows <= 32:
        raise ValueError("max_windows must be between 1 and 32")
    path = Path("/pilot") / f"{sample_id}.avi"
    if not path.is_file():
        raise FileNotFoundError("pilot AVI is not available in the mounted prefix")

    cfg = mmcv.Config.fromfile(CONFIG)
    cfg.model.test_cfg = dict(feature_extraction=True)
    load_start = time.monotonic()
    model = init_recognizer(cfg, "/opt/checkpoints/slowfast.pth", device="cuda:0")
    model.eval()
    load_s = time.monotonic() - load_start

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("could not open pilot AVI")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames < 64 or fps <= 0:
        raise ValueError("video has too few frames or invalid frame rate")

    # The upstream validation recipe samples 32 of 64 frames (interval=2),
    # resizes short edge to 256, center crops 224 and normalizes RGB pixels.
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
                raise RuntimeError(f"unexpected feature shape: {result.shape}")
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
        load_s + decode_s / decoded_frames * (63 + full_windows)
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
        "model_load_s": round(load_s, 3),
        "decode_s": round(decode_s, 3),
        "preprocess_s": round(preprocess_s, 3),
        "gpu_inference_s": round(inference_s, 3),
        "estimated_one_weld_compute_s": round(estimated_compute_s, 1),
        "estimated_one_weld_gpu_usd": round(estimated_compute_s * 0.000306, 3),
        "estimate_excludes": ["image build", "container startup", "CPU/memory", "GCS egress"],
    }


@app.local_entrypoint()
def main(max_windows: int = 16):
    import json

    if any(value.startswith("unset-") for value in (SAMPLE_ID, BUCKET, PREFIX)):
        raise ValueError("set WELDOPS_SAMPLE_ID, WELDOPS_GCS_BUCKET and WELDOPS_GCS_PREFIX")
    if not PREFIX.endswith("/"):
        raise ValueError("WELDOPS_GCS_PREFIX must end with /")
    if not 1 <= max_windows <= 32:
        raise ValueError("max_windows must be between 1 and 32")
    print(json.dumps(benchmark.remote(SAMPLE_ID, max_windows), indent=2))
