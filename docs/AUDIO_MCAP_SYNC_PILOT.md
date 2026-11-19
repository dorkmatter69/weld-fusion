# Audio features, exploratory MCAP, and clock evidence

The one-weld video feature export is complete. The pilot FLAC and CSV were
copied within GCS into the isolated pilot bucket; their byte counts and GCS
MD5 values match the source objects. The source media are not committed to
Git and do not need to download to the Mac.

With the same `WELDOPS_*` shell variables used for the video pilot, run:

```bash
git pull --ff-only
.venv/bin/modal run scripts/modal_audio_mcap_pilot.py
```

The bounded CPU Sandbox downloads only this pilot's AVI, FLAC and CSV through
short-lived signed GET URLs. It writes:

- `features/audio-stft-16k/<sample>.npz`: mono audio at its released 16 kHz
  rate; 1,366-sample Hann windows (85.375 ms), 683-sample hops (42.6875 ms),
  684 frequency bins, `log1p` magnitude, and sample-relative frame centers.
  These time-matched windows are an experimental adaptation. The paper used
  16,384-sample windows at 192 kHz (85.333 ms), which cannot be recovered from
  the released 16 kHz FLAC. No audio autoencoder has been trained yet.
- `mcap/pilot/<sample>.mcap`: `/weld/meta`, `/weld/alignment`,
  `/sensors/weld` (all 335 sensor rows), `/video/brightness` (all 1,119
  frames), and `/audio/rms` (50 ms envelope). Raw AVI and FLAC remain
  separate and are referenced by identity; the MCAP does not contain their
  waveform or video frames. It is an exploratory view with independent
  file-relative clocks, not a verified synchronized log.

Both output objects are create-only and are checked against their GCS size
and MD5 metadata. The worker prints SHA-256 digests. The pilot's category and
train/validation/test split remain `null` until an authoritative manifest
record is audited. Do not infer a Good training label from the file stem.

## Completed pilot run

The user ran the CPU exporter successfully. The audio feature archive has
889 frames × 684 bins, spans 0.042656–37.949156 seconds on the FLAC's own
clock, and is 2,194,781 bytes. Its worker-computed SHA-256 is
`1f292c0f6804c1479452301f207be4c5fa7486078679ae70b2a73f2c90cf1e2d`.
The exploratory MCAP is 83,709 bytes and contains 335 sensor messages,
1,119 video brightness messages, 760 audio RMS messages, and one metadata
and alignment message each. Its SHA-256 is
`499cae728e007a99d4363212e0a947518aa48ac19a4b7626c6737141e74fd28f`.
An independent GCS metadata read confirmed both object sizes and MD5 values.
The exporter returned `clock_alignment_verified: false`, `category: null`,
and `split: null`. These outputs are feature preparation and exploratory
recording, not anomaly detection or validated synchronization.

## Are the streams synchronized?

**No.** The file lengths and candidate arc boundaries are currently:

| Stream | Time basis | Recorded span | Candidate arc-on | Candidate arc-off |
| --- | --- | ---: | ---: | ---: |
| Video | first AVI frame, nominal 31 FPS | 36.097 s | 5.677 s | 30.968 s |
| Audio | first FLAC sample, 16 kHz | 38.000 s | 4.750 s | 30.800 s |
| Sensors | first CSV Date/Time row; timezone unknown | 37.142 s | 6.145 s | 32.129 s |

Relative to video, sensor arc-on differs by +0.468 s and arc-off by +1.161 s.
Audio arc-on differs by −0.927 s and arc-off by −0.168 s. The changing
differences could come from frame-rate error, device start delay, threshold
differences, or more than one of these. The paper notes that video FPS varied
from nominal and used audio event duration as an alignment reference.

For inspection, a two-boundary affine candidate is
`t_sensor = 1.027429 * t_video + 0.311857`, and
`t_audio = 1.030038 * t_video - 1.097959` (seconds from each file's own
start). Two points fit these lines by construction, so they provide **no
independent validation** of the map. They are not applied to MCAP timestamps
or training features.

## Verification needed before a shared clock

1. Inspect current, audio energy and image brightness around both arc
   boundaries. Mark the physical event with an interval of uncertainty,
   rather than assuming the screening threshold identifies the same moment
   in all streams.
2. Check independent landmarks inside the weld and compare residuals under
   an offset-only mapping and an affine offset-plus-rate mapping. Verify
   actual AVI frame timing or a per-frame timestamp source; nominal 31 FPS
   alone is insufficient to establish clock drift.
3. Keep the original audio sample indices, AVI frame indices and CSV Date/Time
   values as provenance. Only then write calibrated mappings, their evidence
   and uncertainty into a new MCAP or episode version. Preserve this
   exploratory version as unaligned.

Whole-weld anomaly score fusion can be evaluated after authoritative labels
are found, even while per-frame clocks remain unresolved. Any per-frame
audio/video/sensor fusion requires the verified mapping first.

## Review artifacts and manifest audit

The pilot plot was generated from the exploratory MCAP by
`scripts/plot_pilot_clock_evidence.py` and stored privately at
`gs://project-da693f7a-9b8c-40dd-a1d-modal-weld-pilot/review/04-01-23-0024-00/clock-evidence-v1.png`.
Each panel starts at its own file's zero; dashed lines mark the screening
thresholds, not ground-truth synchronous events. The image makes the onset
and offset discrepancies visible. No clock model has been applied.

The archive's `raid/intel_robotic_welding_dataset/manifest.csv` was read in a
GCP worker without extracting the 40 GB archive. Its SHA-256 is
`9c4fb4df70bb137fc851dd858ac22d53eed86ba08c3806d7177e40b300406826`
(587,626 bytes). Exactly one row in that manifest matched the pilot sample:
`CATEGORY=Good`, `SPLIT=TRAIN`, `WELD_TYPE=FILLET`, and
`SUBDIRS=2_good_weld_2_02-09-23_Fe410/04-01-23-0024-00`. The private evidence
is `gs://project-da693f7a-9b8c-40dd-a1d-weldops-data/qc/intel-welding/04-01-23-0024-00/manifest-evidence-v1.json`.
This is an authoritative **declared dataset label/split**, not an independent
verification of the physical weld quality. The exploratory MCAP is unchanged
and continues to report an unverified label and unaligned clocks; consumers
should join the audited manifest evidence by the sample ID rather than infer
a label from its filename.

The temporary GCE manifest worker was removed after the result was saved.
The streaming audit inspected the first archive member named `manifest.csv`
and checked for a unique matching row within it; it did not prove that the
archive contains no additional manifest files.

Intel's Manufacturing AI Suite `fusion-analytics/fusion.py` is a useful
reference for **live decision fusion**: it matches MQTT vision and time-series
model outputs by UTC timestamps within a configurable tolerance (50 ms by
default), then applies AND or OR to anomaly decisions. Its vision timestamps
come from RTP sender NTP metadata. Our pilot files have no verified common
clock or online model outputs, so that matcher is not applicable to these
raw recordings yet. The paper's whole-weld audio/video score fusion is a
separate evaluation step.
