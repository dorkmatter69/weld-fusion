"""Export all one-frame-stride SlowFast features for the measured pilot weld."""

import json
import subprocess

import modal

import modal_slowfast_pilot as pilot


app = modal.App("weldops-slowfast-one-weld-export")
OUTPUT_KEY = "features/slowfast-r101-v015/{}.npz".format(pilot.SAMPLE_ID)


def sign_url(object_key: str, verb: str = "GET", headers: str = "") -> str:
    command = [
        "gcloud", "storage", "sign-url",
        "gs://{}/{}".format(pilot.BUCKET, object_key),
        "--duration=1h",
        "--project={}".format(pilot.PROJECT),
        "--region={}".format(pilot.REGION),
        "--impersonate-service-account={}".format(pilot.SIGNER),
        "--http-verb={}".format(verb),
        "--format=json",
    ]
    if headers:
        command.append("--headers={}".format(headers))
    signed = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(signed.stdout)
    row = data[0] if isinstance(data, list) else data
    url = row.get("signed_url") or row.get("signedUrl") or row.get("url")
    if not url or not url.startswith("https://"):
        raise RuntimeError("gcloud did not return a signed HTTPS URL")
    return url


@app.local_entrypoint()
def main():
    if any(value.startswith("unset-") for value in (
            pilot.SAMPLE_ID, pilot.BUCKET, pilot.PROJECT, pilot.SIGNER)):
        raise ValueError("set the WELDOPS_* environment variables from the pilot run")
    if pilot.SAMPLE_ID != "04-01-23-0024-00":
        raise ValueError("export is limited to the measured pilot weld")

    object_uri = "gs://{}/{}".format(pilot.BUCKET, OUTPUT_KEY)
    check = subprocess.run(
        ["gcloud", "storage", "objects", "describe", object_uri, "--format=value(name)"],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        raise RuntimeError("feature object already exists; refusing to overwrite it")
    if "404" not in check.stderr:
        raise RuntimeError("could not verify that the feature object is absent")

    # Build first so all three one-hour signed URLs are fresh for the GPU run.
    pilot.image.build(app)
    get_url = sign_url("pilot/{}.avi".format(pilot.SAMPLE_ID))
    put_url = sign_url(
        OUTPUT_KEY, "PUT",
        "content-type=application/octet-stream,x-goog-if-generation-match=0",
    )
    verify_url = sign_url(OUTPUT_KEY)
    secret = modal.Secret.from_dict({
        "PILOT_SIGNED_URL": get_url,
        "PILOT_PUT_URL": put_url,
        "PILOT_VERIFY_URL": verify_url,
    })
    sandbox = modal.Sandbox.create(
        "python", "/opt/slowfast_pilot_worker.py",
        app=app, image=pilot.image, gpu="A10", cpu=2, memory=8192,
        timeout=600, secrets=[secret],
        env={
            "PILOT_SAMPLE_ID": pilot.SAMPLE_ID,
            "PILOT_EXPORT": "1",
            "PILOT_OUTPUT_OBJECT": OUTPUT_KEY,
        },
    )
    try:
        output = sandbox.stdout.read()
        sandbox.wait()
        if sandbox.returncode != 0:
            raise RuntimeError(
                "feature export failed (exit {}); inspect Modal logs privately".format(
                    sandbox.returncode
                )
            )
        result = pilot.parse_benchmark_output(output)
        if (result["benchmarked_windows"] != 1056
                or result["feature_dimension"] != 2304
                or result["output_object"] != OUTPUT_KEY
                or result["verified_readback"] is not True):
            raise RuntimeError("feature export verification failed")
        print(json.dumps(result, indent=2))
    finally:
        sandbox.terminate()
