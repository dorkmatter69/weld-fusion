# Intel weld data pilot

This workspace contains a one-pass archive inventory/extractor and a per-weld
quality screen. The three-stream pilot QC ran on one private weld; the full
archive inventory and extraction have **not** been executed.
Run them in a machine with adequate disk near the private GCS bucket. Keep the
raw archive and media private and follow the dataset's research-use terms.

## Dataset facts

From the Hugging Face card for `IntelLabs/Intel_Robotic_Welding_Multimodal_Dataset`
(arXiv 2409.02290), not yet confirmed against the private archive:

- Licence: Intel Research Use License Agreement; research only, no commercial use.
- About 39.9 GB across 236 welding sessions. Each session directory holds
  sample directories, each with an AVI, a FLAC, a CSV and `images/` (5 post-weld JPGs).
- Audio is 1 channel at 16 kHz FLAC. Video is a nominal 30 FPS AVI; the card
  does not state resolution.
- A manifest gives each sample's `CATEGORY` (12 classes including `Good`),
  weld type, thickness, steel type, current, voltage, gas, robot speed,
  `DIRECTORY`, `SUBDIRS` and `SPLIT`.
- The card notes that labels were not validated after welding by a welding
  expert, that one weld may contain several defects, and that devices started
  by one script may still be slightly misaligned.

The sensor CSV column names are not listed on the card. The pilot confirmed
`Primary Weld Current` among the numeric sensor columns.

Intel's `industrial-edge-insights-multimodal` weld simulator is not an
alignment reference: it maps CSV rows to video frames by proportion of
duration, replaces the CSV's `Date`/`Time` with publish wall-clock time, and
does not read audio.

## Local setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests/test_archive.py tests/test_recorded_timestamps.py tests/test_episode_export.py
```

If package installation is unavailable, the archive safety tests alone use
only Python's standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_archive.py
```

## First pass: measure without unpacking

Copy the archive to a worker with adequate storage. The scanner reads the
compressed archive once, checks member names and types, hashes decompressed
bytes, and reports total and largest uncompressed member sizes. It does **not**
verify the GCS object's own CRC32C: record the object's generation and checksum
separately before download and verify the full download using cloud tooling.
The scan decompresses and hashes the entire archive even without extraction.
Allow substantial time for the 40 GB read; actual duration depends on disk,
CPU and the archive's contents. Benchmark the worker before scheduling a run.

```bash
python -m weldfusion.archive /data/intel_robotic_welding_dataset.tar.gz \
  /data/inventory.json
```

Inspect `total_uncompressed_bytes`, `largest_member_bytes`, and the manifest
before choosing extraction disk. The optional `--output-dir` writes all
original members to local staging and requires that full space. Do not run it
on Cloud Shell's small home disk. Extraction currently does not upload to GCS,
checkpoint an interrupted pass, or skip verified members; those remain the
next deployment tasks for the full archive. Re-running extraction reads the
entire archive again and atomically **overwrites existing staged files**.

```bash
python -m weldfusion.archive /data/intel_robotic_welding_dataset.tar.gz \
  /data/inventory.json --output-dir /data/staged
```

## One weld, eight files

The observed pilot has one CSV, one AVI, one FLAC and five JPG images: eight
files belonging to **one weld**, `04-01-23-0024-00`. Locate its directory under
`/data/staged` using the archive-relative path recorded in `inventory.json`.
The directory basename must match the CSV, AVI and FLAC basenames because the
QC command constructs filenames from the directory name. Per the dataset card,
samples sit one level below a session directory. For example, if the archive has
`<session>/04-01-23-0024-00/04-01-23-0024-00.csv`, run:

```bash
python -m weldfusion.qc /data/staged/<session>/04-01-23-0024-00 \
  /data/qc/04-01-23-0024-00.json
```

The JSON records media duration, row/frame/sample counts, candidate active
interval starts and ends, and the difference between sensor and video candidate
events. These are screening results on **separate file-relative clocks**.
The QC command does not read or validate the five post-weld JPGs.
`status` stays `unresolved` and `clock_alignment_verified` stays false until
independent anchors and a human reviewer justify a mapping. The JPG filenames
are not synchronized timestamps. The tool returns a nonzero exit code if any
of the three main streams is absent or unreadable, while still writing QC JSON.

Next, sample across labels and acquisition groups, review candidate arc-on and
arc-off events, and benchmark worker duration and cost before a full run. Draw
the sample from the manifest's `CATEGORY`, `WELD_TYPE`, session `DIRECTORY` and
`SPLIT` columns. Because labels are unvalidated and may hide multiple defects,
do not treat stream agreement as label confirmation. The full GCS
extraction/job orchestration has not been deployed.

## If Cloud Shell is unavailable: Compute Engine pilot

`scripts/build_gce_pilot_startup.py` generates a startup script containing the
four local Python modules needed for QC. It does not embed credentials or media.
A temporary Debian 12 VM in `us-west1-b` can run that script using the existing
`weldops-transfer` service account. That account has object viewer and creator
roles on the private bucket. The VM downloads only the three pilot streams to
its own boot disk, installs Python dependencies, runs QC, and creates
`gs://project-da693f7a-9b8c-40dd-a1d-weldops-data/qc/intel-welding/04-01-23-0024-00/qc-v1.json`.
It refuses to overwrite that output. No private media is copied to a laptop.

Building and syntax-checking the script are local and free:

```bash
python3 scripts/build_gce_pilot_startup.py /tmp/weldops-pilot-startup.sh
bash -n /tmp/weldops-pilot-startup.sh
```

Creating the VM starts billing and executes the private-data job. Provisioning
requires the project account to have Compute Engine instance creation and
service-account-use permissions. The VM also needs outbound access to Debian
and PyPI for `python3-venv`, NumPy, OpenCV, SoundFile and Google Cloud Storage's
Python client. It should be deleted after the QC JSON or startup error has been
reviewed. The pilot ran on 2026-11-18 and the VM was deleted after it wrote
the JSON. It found 335 sensor rows spanning 37.142 seconds, 1,119 video frames
at 31 FPS (36.097 seconds), and 608,000 mono audio samples at 16 kHz
(38 seconds). Candidate arc-on times were 6.145 seconds in sensor current,
5.677 seconds in video brightness, and 4.750 seconds in audio energy. These
are separate file-relative clocks; the result remains `unresolved`, with zero
warnings. The report remains in the private bucket at the path above.
