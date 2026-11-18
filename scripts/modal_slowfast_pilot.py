"""Sign one pilot AVI locally, then benchmark in a legacy-Python GPU Sandbox."""

import json
import os
import subprocess
from pathlib import Path

import modal


SAMPLE_ID = os.environ.get("WELDOPS_SAMPLE_ID", "unset-sample")
BUCKET = os.environ.get("WELDOPS_GCS_BUCKET", "unset-bucket")
PROJECT = os.environ.get("WELDOPS_GCP_PROJECT", "unset-project")
SIGNER = os.environ.get("WELDOPS_SIGNING_SERVICE_ACCOUNT", "unset-signer")
REGION = os.environ.get("WELDOPS_GCS_REGION", "us-west1")
CHECKPOINT = (
    "https://download.openmmlab.com/mmaction/recognition/slowfast/"
    "slowfast_r101_4x16x1_256e_kinetics400_rgb/"
    "slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth"
)

# Sandboxes can run old Python/PyTorch without adding Python through Modal's
# current image builder (which only offers Python 3.11+).
image = (
    modal.Image.from_registry("pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime")
    .apt_install("git", "curl", "libgl1", "libglib2.0-0", "ca-certificates")
    .run_commands(
        "python -m pip install 'numpy==1.23.5' 'opencv-python-headless==4.5.5.64'",
        "python -m pip install 'mmcv-full==1.4.0' "
        "-f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html",
        "git clone --depth 1 --branch v0.15.0 "
        "https://github.com/open-mmlab/mmaction2.git /opt/mmaction2",
        "python -m pip install --no-deps -e /opt/mmaction2",
        "mkdir -p /opt/checkpoints && curl -fL --retry 3 "
        f"'{CHECKPOINT}' -o /opt/checkpoints/slowfast.pth",
    )
    .add_local_file(
        str(Path(__file__).with_name("slowfast_pilot_worker.py")),
        "/opt/slowfast_pilot_worker.py",
        copy=True,
    )
)

app = modal.App("weldops-slowfast-one-weld-benchmark")


def expected_windows(frame_count: int) -> int:
    """Number of full 64-frame windows at one-frame stride."""
    return max(0, frame_count - 63)


@app.local_entrypoint()
def main(max_windows: int = 16):
    if any(value.startswith("unset-") for value in (SAMPLE_ID, BUCKET, PROJECT, SIGNER)):
        raise ValueError(
            "set WELDOPS_SAMPLE_ID, WELDOPS_GCS_BUCKET, WELDOPS_GCP_PROJECT, "
            "and WELDOPS_SIGNING_SERVICE_ACCOUNT"
        )
    if not 1 <= max_windows <= 32:
        raise ValueError("max_windows must be between 1 and 32")

    # Build before signing: old dependency downloads can exceed URL lifetime.
    image.build(app)
    command = [
        "gcloud", "storage", "sign-url",
        f"gs://{BUCKET}/pilot/{SAMPLE_ID}.avi",
        "--duration=1h", f"--project={PROJECT}", f"--region={REGION}",
        f"--impersonate-service-account={SIGNER}", "--format=json",
    ]
    signed = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(signed.stdout)
    row = data[0] if isinstance(data, list) else data
    url = row.get("signed_url") or row.get("signedUrl") or row.get("url")
    if not url or not url.startswith("https://"):
        raise RuntimeError("gcloud did not return a signed HTTPS URL")

    # Keep the short-lived bearer URL out of the command line and Modal logs.
    secret = modal.Secret.from_dict({"PILOT_SIGNED_URL": url})
    sandbox = modal.Sandbox.create(
        "python", "/opt/slowfast_pilot_worker.py",
        app=app,
        image=image,
        gpu="A10",
        cpu=2,
        memory=8192,
        timeout=600,
        secrets=[secret],
        env={"PILOT_SAMPLE_ID": SAMPLE_ID, "PILOT_MAX_WINDOWS": str(max_windows)},
    )
    try:
        output = sandbox.stdout.read()
        sandbox.wait()
        status = sandbox.returncode
        if status != 0:
            raise RuntimeError(
                "SlowFast Sandbox failed (exit {}); inspect Modal build and runtime logs privately".format(status)
            )
        result = json.loads(output)
        if result["benchmarked_windows"] > max_windows or result["feature_dimension"] != 2304:
            raise RuntimeError("Sandbox returned an invalid benchmark result")
        print(json.dumps(result, indent=2))
    finally:
        sandbox.terminate()
