"""Read the Intel archive's manifest in a stream and audit one weld record.

No archive member is extracted. The JSON output contains only the matching
manifest row and provenance, never the full dataset manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path


def audit(stream, sample_id: str) -> dict:
    candidates = []
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and Path(member.name).name.lower() == "manifest.csv":
                if member.size > 20_000_000:
                    raise ValueError("manifest exceeds 20 MB safety limit")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("manifest cannot be read")
                content = extracted.read(20_000_001)
                if len(content) != member.size:
                    raise ValueError("truncated manifest")
                reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
                if not reader.fieldnames or not {"SUBDIRS", "CATEGORY", "SPLIT"}.issubset(reader.fieldnames):
                    raise ValueError("manifest lacks required columns")
                rows = [row for row in reader if row["SUBDIRS"].strip("/").split("/")[-1] == sample_id]
                candidates.append({
                    "manifest_member": member.name,
                    "manifest_sha256": hashlib.sha256(content).hexdigest(),
                    "manifest_bytes": len(content),
                    "matching_rows": rows,
                })
                break
    if len(candidates) != 1 or len(candidates[0]["matching_rows"]) != 1:
        raise ValueError(f"expected one manifest and one {sample_id} row; found {len(candidates)} manifests and {sum(len(c['matching_rows']) for c in candidates)} rows")
    evidence = candidates[0]
    return {
        "sample_id": sample_id,
        "manifest_member": evidence["manifest_member"],
        "manifest_sha256": evidence["manifest_sha256"],
        "manifest_bytes": evidence["manifest_bytes"],
        "record": evidence["matching_rows"][0],
        "record_status": "matched_single_manifest_row",
        "stream_clock_alignment_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-id", default="04-01-23-0024-00")
    args = parser.parse_args()
    with args.archive.open("rb") as source:
        result = audit(source, args.sample_id)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
