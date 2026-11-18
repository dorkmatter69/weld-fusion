"""One-pass safe tar inventory and optional extraction to a staging directory.

Usage: python -m weldfusion.archive ARCHIVE.tar.gz REPORT.json [--output-dir STAGING]
The archive is streamed, never unpacked as a whole. Staging must have enough
space for the uncompressed data; leave --output-dir off to measure it first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def _safe_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"unsafe archive member: {name!r}")
    return path.as_posix()


def inventory(archive: str | Path, output_dir: str | Path | None = None) -> dict:
    """Reject nonregular entries and duplicates; hash every decompressed member."""
    output = Path(output_dir).resolve() if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    seen, members = set(), []
    total = largest = 0
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            name = _safe_name(item.name)
            if item.isdir():
                continue
            if not item.isfile():
                raise ValueError(f"nonregular archive member: {name}")
            if name in seen:
                raise ValueError(f"duplicate archive member: {name}")
            seen.add(name)
            source = tar.extractfile(item)
            if source is None:
                raise ValueError(f"cannot read archive member: {name}")
            dest = output / name if output is not None else None
            # Atomic replacement prevents partial files from looking complete.
            tmp = None
            digest = hashlib.sha256()
            size = 0
            try:
                if dest is not None:
                    relative_parts = PurePosixPath(name).parts
                    paths = [output.joinpath(*relative_parts[:i]) for i in range(1, len(relative_parts) + 1)]
                    if any(p.is_symlink() for p in paths):
                        raise ValueError(f"symlink in output path: {dest}")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".weldops-")
                    writer = os.fdopen(fd, "wb")
                else:
                    writer = None
                try:
                    while chunk := source.read(1024 * 1024):
                        size += len(chunk)
                        digest.update(chunk)
                        if writer is not None:
                            writer.write(chunk)
                finally:
                    if writer is not None:
                        writer.close()
                if size != item.size:
                    raise ValueError(f"size mismatch for {name}: {size} != {item.size}")
                if dest is not None:
                    os.replace(tmp, dest)
                    tmp = None
            finally:
                if tmp is not None:
                    os.unlink(tmp)
            total += size
            largest = max(largest, size)
            members.append({"archive_path": name, "size_bytes": size,
                            "sha256": digest.hexdigest()})
    return {"archive": str(archive), "member_count": len(members),
            "total_uncompressed_bytes": total, "largest_member_bytes": largest,
            "members": members}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    report = inventory(args.archive, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['member_count']} members; {report['total_uncompressed_bytes']} uncompressed bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
