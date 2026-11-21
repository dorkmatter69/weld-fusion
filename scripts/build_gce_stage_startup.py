"""Build a Compute Engine startup script that unpacks the Intel archive inside GCP.

    python3 scripts/build_gce_stage_startup.py /tmp/weldops-stage-startup.sh

The VM copies the archive from the private bucket to its own disk, extracts it
with weldfusion.archive (path checks, per-member SHA-256), and uploads the tree
to gs://BUCKET/staged/intel-welding/ plus an inventory to
gs://BUCKET/qc/intel-welding/full/. It refuses to run if the staged prefix
already holds objects. Media never leaves GCP. The script embeds code only.
"""

from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "project-da693f7a-9b8c-40dd-a1d-weldops-data"
SOURCE = f"gs://{BUCKET}/raw/intel-welding/intel_robotic_welding_dataset.tar.gz"
STAGED = f"gs://{BUCKET}/staged/intel-welding"
QC = f"gs://{BUCKET}/qc/intel-welding/full"
ARCHIVE_ROOT = "raid/intel_robotic_welding_dataset"


def build() -> str:
    code = base64.b64encode((ROOT / "src/weldfusion/archive.py").read_bytes()).decode()
    return f"""#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/weldops-stage.log) 2>&1
echo WELDOPS_STAGE_START
if [ -n "$(gcloud storage ls '{STAGED}/' 2>/dev/null | head -1)" ]; then
  echo 'WELDOPS_STAGE_FAILED staged prefix is not empty'; exit 1
fi
mkdir -p /opt/weldops/weldfusion /data
touch /opt/weldops/weldfusion/__init__.py
base64 -d > /opt/weldops/weldfusion/archive.py <<'CODE'
{code}
CODE
gcloud storage cp '{SOURCE}' /data/archive.tar.gz
echo WELDOPS_STAGE_DOWNLOADED
PYTHONPATH=/opt/weldops python3 -m weldfusion.archive /data/archive.tar.gz \\
  /data/inventory-v1.json --output-dir /data/staged
rm /data/archive.tar.gz
echo WELDOPS_STAGE_EXTRACTED
test -f '/data/staged/{ARCHIVE_ROOT}/manifest.csv'
gcloud storage rsync -r '/data/staged/{ARCHIVE_ROOT}' '{STAGED}'
members=$(python3 -c "import json;r=json.load(open('/data/inventory-v1.json'));print(sum(m['archive_path'].startswith('{ARCHIVE_ROOT}/') for m in r['members']), r['member_count'])")
objects=$(gcloud storage ls -r '{STAGED}/**' | wc -l)
echo "WELDOPS_STAGE_COUNTS members_under_root/total=$members objects=$objects"
gcloud storage cp --no-clobber /data/inventory-v1.json '{QC}/inventory-v1.json'
echo "WELDOPS_STAGE_DONE $objects"
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build())
    print(f"Wrote {args.output} (code only, {args.output.stat().st_size} bytes)")
