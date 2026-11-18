# Intel weld data pilot

This workspace contains a one-pass archive inventory/extractor and a per-weld
quality screen. They have **not** been executed on the private archive.
Run them in a machine with adequate disk near the private GCS bucket. Keep the
raw archive and media private and follow the dataset's research-use terms.

## Local setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/test_archive.py tests/test_recorded_timestamps.py tests/test_episode_export.py
```

## First pass: measure without unpacking

Copy the archive to a worker with adequate storage. The scanner reads the
compressed archive once, checks member names and types, hashes decompressed
bytes, and reports total and largest uncompressed member sizes. It does **not**
verify the GCS object's own CRC32C: record the object's generation and checksum
separately before download and verify the full download using cloud tooling.

```bash
python -m weldfusion.archive /data/intel_robotic_welding_dataset.tar.gz \
  /data/inventory.json
```

Inspect `total_uncompressed_bytes`, `largest_member_bytes`, and the manifest
before choosing extraction disk. The optional `--output-dir` writes all
original members to local staging and requires that full space. Do not run it
on Cloud Shell's small home disk. Extraction currently does not upload to GCS,
checkpoint an interrupted pass, or skip verified members; those remain the
next deployment tasks for the full archive.

```bash
python -m weldfusion.archive /data/intel_robotic_welding_dataset.tar.gz \
  /data/inventory.json --output-dir /data/staged
```

## Eight-file pilot

The extracted directory contains `<id>.csv`, `<id>.avi`, `<id>.flac` and
`images/*.jpg`. Run:

```bash
python -m weldfusion.qc /data/04-01-23-0024-00 /data/qc/04-01-23-0024-00.json
```

The JSON records media duration, row/frame/sample counts, candidate active
interval starts and ends, and the difference between sensor and video candidate
events. These are screening results on **separate file-relative clocks**.
`status` stays `unresolved` and `clock_alignment_verified` stays false until
independent anchors and a human reviewer justify a mapping. The JPG filenames
are not synchronized timestamps. The tool returns a nonzero exit code if any
of the three main streams is absent or unreadable, while still writing QC JSON.

Next, sample across labels and acquisition groups, review candidate arc-on and
arc-off events, and benchmark worker duration and cost before a full run. The
full GCS extraction/job orchestration has not been deployed.
