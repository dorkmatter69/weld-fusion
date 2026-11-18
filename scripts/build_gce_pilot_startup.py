"""Build a Compute Engine startup script containing the local QC source.

The generated script reads one private GCS pilot in the VM and writes only QC
JSON back to the same bucket. It embeds code, never credentials or media.
"""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__.py", "dataset.py", "align.py", "qc.py")
BUCKET = "project-da693f7a-9b8c-40dd-a1d-weldops-data"
PILOT = "04-01-23-0024-00"


def build() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for module in MODULES:
            path = ROOT / "src" / "weldfusion" / module
            archive.add(path, arcname=f"weldfusion/{module}")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"""#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/weldops-pilot.log) 2>&1
echo 'WELDOPS_PILOT_START'
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv
python3 -m venv /opt/weldops-venv
/opt/weldops-venv/bin/pip install --disable-pip-version-check \\
  'numpy>=1.24' 'opencv-python-headless>=4.8' 'soundfile>=0.12' 'google-cloud-storage>=2'
mkdir -p /opt/weldops /var/lib/weldops/{PILOT}
base64 -d > /opt/weldops-source.tar.gz <<'SOURCE'
{encoded}
SOURCE
tar -xzf /opt/weldops-source.tar.gz -C /opt/weldops
/opt/weldops-venv/bin/python - <<'PY'
from google.cloud import storage
from pathlib import Path
bucket = storage.Client().bucket({BUCKET!r})
prefix = 'pilot/intel-welding/{PILOT}/'
root = Path('/var/lib/weldops/{PILOT}')
expected = {{{PILOT!r} + ext for ext in ('.csv', '.avi', '.flac')}}
for name in expected:
    blob = bucket.blob(prefix + name)
    if not blob.exists():
        raise FileNotFoundError(prefix + name)
    blob.download_to_filename(root / name)
print('Pilot streams downloaded inside GCP:', sorted(expected), flush=True)
PY
PYTHONPATH=/opt/weldops /opt/weldops-venv/bin/python -m weldfusion.qc \\
  /var/lib/weldops/{PILOT} /var/lib/weldops/{PILOT}/qc-v1.json
/opt/weldops-venv/bin/python - <<'PY'
from google.cloud import storage
bucket = storage.Client().bucket({BUCKET!r})
name = 'qc/intel-welding/{PILOT}/qc-v1.json'
blob = bucket.blob(name)
blob.upload_from_filename('/var/lib/weldops/{PILOT}/qc-v1.json',
                          content_type='application/json', if_generation_match=0)
print('WELDOPS_PILOT_DONE gs://' + bucket.name + '/' + name, flush=True)
PY
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build())
    print(f"Wrote {args.output} (code only, {args.output.stat().st_size} bytes)")
