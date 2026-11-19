"""Build a bounded GCE startup job to inspect the private archive manifest."""

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "project-da693f7a-9b8c-40dd-a1d"
BUCKET = PROJECT + "-weldops-data"
SOURCE = "raw/intel-welding/intel_robotic_welding_dataset.tar.gz"
OUTPUT = "qc/intel-welding/04-01-23-0024-00/manifest-evidence-v1.json"


def build() -> str:
    code = base64.b64encode((ROOT / "scripts/audit_archive_manifest.py").read_bytes()).decode()
    return f"""#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/weldops-manifest.log) 2>&1
echo WELDOPS_MANIFEST_START
base64 -d > /opt/audit_archive_manifest.py <<'CODE'
{code}
CODE
timeout 3600s python3 - <<'PY'
import importlib.util
import json
import urllib.parse
import urllib.request

spec = importlib.util.spec_from_file_location('audit_archive_manifest', '/opt/audit_archive_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
metadata = urllib.request.Request(
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
    headers={{'Metadata-Flavor': 'Google'}})
with urllib.request.urlopen(metadata, timeout=20) as r:
    token = json.load(r)['access_token']
headers = {{'Authorization': 'Bearer ' + token}}
source = 'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/' + urllib.parse.quote({SOURCE!r}, safe='') + '?alt=media'
with urllib.request.urlopen(urllib.request.Request(source, headers=headers), timeout=3600) as stream:
    result = module.audit(stream, '04-01-23-0024-00')
result['archive_object'] = 'gs://{BUCKET}/{SOURCE}'
body = (json.dumps(result, indent=2) + '\\n').encode()
destination = ('https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o?uploadType=media&name='
               + urllib.parse.quote({OUTPUT!r}, safe='') + '&ifGenerationMatch=0')
request = urllib.request.Request(destination, data=body, headers={{**headers, 'Content-Type': 'application/json'}}, method='POST')
with urllib.request.urlopen(request, timeout=60) as r:
    uploaded = json.load(r)
print('WELDOPS_MANIFEST_DONE ' + uploaded['name'], flush=True)
PY
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build())
    print("Wrote manifest startup script (source code only)")
