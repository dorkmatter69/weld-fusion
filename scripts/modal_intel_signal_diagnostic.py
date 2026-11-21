"""Sign the Intel pilot CSV locally; summarize its electrical readings in a Modal Sandbox.

    .venv/bin/modal run scripts/modal_intel_signal_diagnostic.py

Writes ~/weldops-private/intel/<sample>-signal-diagnostic.json. Nothing is uploaded.
"""

import json
from pathlib import Path

import modal

import modal_audio_mcap_pilot as signing
import modal_slowfast_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "weldops-private" / "intel"

app = modal.App("weldops-intel-signal-diagnostic")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26,<3")
    .add_local_dir(str(ROOT / "src" / "weldfusion"), "/opt/weldfusion", copy=True,
                  ignore=["**/__pycache__/**"])
    .add_local_file(str(ROOT / "scripts" / "timestral_pilot_worker.py"),
                    "/opt/timestral_pilot_worker.py", copy=True)
    .add_local_file(str(ROOT / "scripts" / "intel_signal_diagnostic_worker.py"),
                    "/opt/intel_signal_diagnostic_worker.py", copy=True)
)


@app.local_entrypoint()
def main():
    if any(value.startswith("unset-") for value in
           (pilot.SAMPLE_ID, pilot.BUCKET, pilot.PROJECT, pilot.SIGNER)):
        raise ValueError("set the WELDOPS_* variables for the measured pilot")
    image.build(app)
    secrets = {"PILOT_GET_CSV": signing.sign_url("pilot/{}.csv".format(pilot.SAMPLE_ID))}
    sandbox = modal.Sandbox.create(
        "python", "/opt/intel_signal_diagnostic_worker.py",
        app=app, image=image, cpu=1, memory=2048, timeout=300,
        secrets=[modal.Secret.from_dict(secrets)],
        env={"PYTHONPATH": "/opt", "PILOT_SAMPLE_ID": pilot.SAMPLE_ID},
    )
    try:
        output = sandbox.stdout.read()
        errors = sandbox.stderr.read()
        sandbox.wait()
        if sandbox.returncode != 0:
            tail = [line for line in errors.splitlines() if line.strip()][-1:]
            raise RuntimeError("diagnostic worker failed (exit {}): {}".format(
                sandbox.returncode, tail[0] if tail else "no stderr"))
        records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        if len(records) != 1:
            raise RuntimeError("expected one diagnostic record")
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "{}-signal-diagnostic.json".format(pilot.SAMPLE_ID)
        path.write_text(json.dumps(records[0], indent=2) + "\n")
        print(json.dumps(records[0], indent=2))
        print("Wrote {}".format(path))
    finally:
        sandbox.terminate()
