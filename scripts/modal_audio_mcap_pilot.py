"""Run the bounded 16 kHz audio and exploratory MCAP pilot without local media."""

import json
from pathlib import Path
import subprocess

import modal

import modal_slowfast_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("weldops-audio-mcap-one-weld-pilot")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy==1.26.4", "soundfile==0.12.1",
        "opencv-python-headless==4.10.0.84", "mcap==1.3.0",
    )
    .add_local_dir(str(ROOT / "src" / "weldfusion"), "/opt/weldfusion", copy=True)
    .add_local_file(str(ROOT / "scripts" / "audio_mcap_pilot_worker.py"),
                    "/opt/audio_mcap_pilot_worker.py", copy=True)
)
AUDIO_KEY = "features/audio-stft-16k/{}.npz".format(pilot.SAMPLE_ID)
MCAP_KEY = "mcap/pilot/{}.mcap".format(pilot.SAMPLE_ID)


def sign_url(object_key, verb="GET", headers=None):
    command = [
        "gcloud", "storage", "sign-url",
        "gs://{}/{}".format(pilot.BUCKET, object_key),
        "--duration=1h", "--project={}".format(pilot.PROJECT),
        "--region={}".format(pilot.REGION),
        "--impersonate-service-account={}".format(pilot.SIGNER),
        "--http-verb={}".format(verb), "--format=json",
    ]
    if headers:
        command.append("--headers={}".format(headers))
    signed = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(signed.stdout)
    row = data[0] if isinstance(data, list) else data
    url = row.get("signed_url") or row.get("signedUrl") or row.get("url")
    if not url or not url.startswith("https://"):
        raise RuntimeError("gcloud did not return a signed URL")
    return url


def object_uri(key):
    return "gs://{}/{}".format(pilot.BUCKET, key)


def require_absent(key):
    result = subprocess.run(
        ["gcloud", "storage", "objects", "describe", object_uri(key),
         "--format=value(name)"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        raise RuntimeError("output already exists: {}".format(key))
    if "404" not in result.stderr:
        raise RuntimeError("could not verify absence of {}".format(key))


@app.local_entrypoint()
def main():
    if pilot.SAMPLE_ID != "04-01-23-0024-00" or any(
            value.startswith("unset-") for value in
            (pilot.BUCKET, pilot.PROJECT, pilot.SIGNER)):
        raise ValueError("set the WELDOPS_* variables for the measured pilot")
    for key in (AUDIO_KEY, MCAP_KEY):
        require_absent(key)
    image.build(app)
    put_headers = "content-type=application/octet-stream,x-goog-if-generation-match=0"
    secrets = {
        "PILOT_GET_" + ext.upper(): sign_url("pilot/{}.{}".format(pilot.SAMPLE_ID, ext))
        for ext in ("avi", "flac", "csv")
    }
    secrets["PILOT_PUT_AUDIO"] = sign_url(AUDIO_KEY, "PUT", put_headers)
    secrets["PILOT_PUT_MCAP"] = sign_url(MCAP_KEY, "PUT", put_headers)
    sandbox = modal.Sandbox.create(
        "python", "/opt/audio_mcap_pilot_worker.py",
        app=app, image=image, cpu=2, memory=4096, timeout=600,
        secrets=[modal.Secret.from_dict(secrets)],
        env={"PILOT_SAMPLE_ID": pilot.SAMPLE_ID,
             "PILOT_AUDIO_OBJECT": AUDIO_KEY,
             "PILOT_MCAP_OBJECT": MCAP_KEY},
    )
    try:
        output = sandbox.stdout.read()
        sandbox.wait()
        if sandbox.returncode != 0:
            raise RuntimeError(
                "audio/MCAP worker failed (exit {}); inspect Modal logs privately".format(
                    sandbox.returncode
                )
            )
        records = []
        for line in output.splitlines():
            if line.startswith("{"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if len(records) != 1:
            raise RuntimeError("expected one audio/MCAP result record")
        result = records[0]
        if result["clock_alignment_verified"] is not False:
            raise RuntimeError("unexpected alignment claim")
        if {a["object"] for a in result["artifacts"]} != {AUDIO_KEY, MCAP_KEY}:
            raise RuntimeError("unexpected artifact set")
        for artifact in result["artifacts"]:
            metadata = subprocess.run(
                ["gcloud", "storage", "objects", "describe",
                 object_uri(artifact["object"]), "--format=json"],
                capture_output=True, text=True, check=True,
            )
            stored = json.loads(metadata.stdout)
            if (str(stored.get("size")) != str(artifact["bytes"])
                    or stored.get("md5_hash") != artifact["md5_base64"]):
                raise RuntimeError("GCS checksum mismatch: {}".format(artifact["object"]))
        result["verified_gcs_checksums"] = True
        print(json.dumps(result, indent=2))
    finally:
        sandbox.terminate()
