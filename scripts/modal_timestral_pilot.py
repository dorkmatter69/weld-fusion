"""Sign the pilot sensor CSV locally, then compare Timestral-5M and TimesFM 3.0 in a Modal CPU Sandbox.

Usage (WELDOPS_* variables set as for the other pilots):

    .venv/bin/modal run scripts/modal_timestral_pilot.py --dry-run
    .venv/bin/modal run scripts/modal_timestral_pilot.py

The report is written to ~/weldops-private/timestral/ (outside Git). Nothing is
uploaded to GCS; the Sandbox only receives a one-hour GET URL for one CSV.
"""

import json
from pathlib import Path

import modal

import modal_audio_mcap_pilot as signing
import modal_slowfast_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
TIMESTRAL_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
TIMESFM_VERSION = "3.0.2"   # TimesFM 3.0 weights: non-commercial, non-production licence
DEFAULT_COLUMNS = "Primary Weld Current,Secondary Weld Voltage"
REPORT_DIR = Path.home() / "weldops-private" / "timestral"

app = modal.App("weldops-timestral-one-weld-pilot")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("numpy>=1.26,<3", "torch>=2.4",
                 extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("git+https://github.com/mistralai/timestral.git@{}".format(TIMESTRAL_COMMIT),
                 "timesfm=={}".format(TIMESFM_VERSION))
    .env({"HF_HOME": "/opt/hf"})
    # Bake the public weights into the image so the Sandbox fetches only the CSV.
    .run_commands("python -c \"from timestral import TimestralForecaster; "
                  "TimestralForecaster.from_pretrained('mistralai/Timestral-5M')\"",
                  "python -c \"from timesfm3 import TimesFM3Forecaster; "
                  "TimesFM3Forecaster.from_pretrained('google/timesfm-3.0-pytorch', device='cpu')\"")
    .add_local_dir(str(ROOT / "src" / "weldfusion"), "/opt/weldfusion", copy=True,
                  ignore=["**/__pycache__/**"])
    .add_local_file(str(ROOT / "scripts" / "timestral_pilot_worker.py"),
                    "/opt/timestral_pilot_worker.py", copy=True)
)


@app.local_entrypoint()
def main(columns: str = DEFAULT_COLUMNS, context: int = 176, horizon: int = 32,
         models: str = "timestral,timesfm", cadences: str = "strict,rows,resample",
         window: str = "arc_on", dry_run: bool = False):
    if any(value.startswith("unset-") for value in
           (pilot.SAMPLE_ID, pilot.BUCKET, pilot.PROJECT, pilot.SIGNER)):
        raise ValueError("set the WELDOPS_* variables for the measured pilot")
    selected = [name.strip() for name in columns.split(",") if name.strip()]
    image.build(app)
    secrets = {"PILOT_GET_CSV": signing.sign_url("pilot/{}.csv".format(pilot.SAMPLE_ID))}
    sandbox = modal.Sandbox.create(
        "python", "/opt/timestral_pilot_worker.py",
        app=app, image=image, cpu=4, memory=8192, timeout=900,
        secrets=[modal.Secret.from_dict(secrets)],
        env={"PYTHONPATH": "/opt", "PILOT_SAMPLE_ID": pilot.SAMPLE_ID,
             "PILOT_COLUMNS": json.dumps(selected), "PILOT_CONTEXT": str(context),
             "PILOT_HORIZON": str(horizon), "PILOT_DRY_RUN": "1" if dry_run else "0",
             "PILOT_MODELS": json.dumps([m.strip() for m in models.split(",") if m.strip()]),
             "PILOT_CADENCES": json.dumps([c.strip() for c in cadences.split(",") if c.strip()]),
             "PILOT_WINDOW": window},
    )
    try:
        output = sandbox.stdout.read()
        errors = sandbox.stderr.read()
        sandbox.wait()
        if sandbox.returncode != 0:
            # Worker errors never contain the signed URL; show the last line only.
            tail = [line for line in errors.splitlines() if line.strip()][-1:]
            raise RuntimeError("Timestral worker failed (exit {}): {}".format(
                sandbox.returncode, tail[0] if tail else "no stderr"))
        records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        if len(records) != 1:
            raise RuntimeError("expected one Timestral result record")
        report = records[0]
        if report["cross_stream_alignment_verified"] is not False:
            raise RuntimeError("unexpected alignment claim")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / "{}-{}.json".format(
            pilot.SAMPLE_ID, "preflight" if dry_run else "forecast-comparison")
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print("Wrote {}".format(path))
    finally:
        sandbox.terminate()
