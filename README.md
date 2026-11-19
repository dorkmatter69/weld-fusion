# weldfusion

## WeldOps integration prototype

This branch now carries the original `weld_twin` MuJoCo flat-coupon simulator
from the unversioned `~/Developer/weld-twin` directory alongside the Intel
loader and MCAP converter. The simulator's source and tests were copied without
changing that original directory. Install `.[sim]` for the MuJoCo/pyads
dependencies; the browser demo never imports or connects to `AdsPLC`.

Run the browser demo from the repository root:

    python3 -m http.server 8765 --directory web

Open `http://localhost:8765`. Its built-in animation offers scripted synthetic
telemetry, virtual PLC states, four fault injections, and three separate deterministic reviews with
an adjudication summary. These are rules, not trained agents or physical weld
quality decisions. Use the file picker to load a `WeldEpisode` JSON file.

To load a trace from the actual copied MuJoCo and Python `MockPLC` implementation:

    python -m weldfusion.sim_export mujoco-episode.json --seed 0

Then open `mujoco-episode.json` with the browser file picker. This offline bridge
preserves the Python simulator's distinct behavior; the built-in JavaScript
fault animation is not the MuJoCo or Python PLC implementation.

The shared JSON shape is `contract/weld-episode.schema.json`. To export the
private Intel pilot after obtaining its files and dataset `manifest.csv`:

    python -m weldfusion.episode_export /path/to/04-01-23-0024-00 pilot.json \
      --manifest /path/to/manifest.csv \
      --gcs-prefix gs://project-da693f7a-9b8c-40dd-a1d-weldops-data/pilot/intel-welding/04-01-23-0024-00

Recorded episodes preserve measured sensor intervals and explicitly mark
camera, audio, sensor, and post-weld image clock alignment as unverified. The
browser displays the imported sensor telemetry but does not fetch private GCS
media; that requires an authenticated backend. No Intel data is checked in.

To make the original `~/Developer/weld-twin` directory canonical after review,
run `bash scripts/migrate_canonical.sh` from this staged repository on the
same Mac. It first commits the original simulator on `legacy/simulator` in
that directory, then checks out the existing GitHub `feat/virtual-cell-mvp`
history and applies this integration commit. It refuses an already versioned
directory or a remote branch that has moved. The script has not been run on
the original directory from this restricted workspace.

The Intel loader joins full-dataset `manifest.csv` rows by subdirectory and
keeps same-named sample IDs in different directories separate, including
distinct MCAP filenames. The manifest's category text is retained; non-Good
defect status/class IDs stay unknown until the full label vocabulary is audited.
Full-dataset
conversion has not been run. MCAP conversion refuses timestamped records by
default because sensor and camera time origins are unregistered. Explicit
`--allow-unregistered-clocks` permits exploratory output but still does not
calibrate clocks. Do not interpret its event offset as a device-clock offset.

Package 1 of the multimodal weld-defect case study: find the welds in the
Intel Robotic Welding Multimodal Dataset, put each weld's video, sensor and
audio streams on one timeline, and write one MCAP file per weld.

## Install

    pip install -e ".[dev]"
    pytest

## Use

    weldfusion scan     /path/to/dataset            # classes, modalities, sensor columns
    weldfusion convert  /path/to/dataset  out/      # one MCAP per weld + manifest.json
    weldfusion inspect  out/<stem>.mcap             # topics, counts, alignment

Useful `convert` options:

- `--embed-every N` embeds every Nth frame as JPEG so the weld can be scrubbed
  in Foxglove. Without it the MCAP holds signals only and stays small.
- `--sensor-hz R` sets the sensor rate instead of inferring it.
- `--limit N` converts only the first N welds.
- `--allow-unregistered-clocks` permits exploratory MCAP creation from real
  timestamped sensor files; their camera and sensor origins remain unaligned.

To try it without the dataset:

    python -c "from weldfusion.synth import make_sample; make_sample('demo')"
    weldfusion convert demo demo_out --embed-every 15

## What is in each MCAP

| Topic | Content |
|---|---|
| `/weld/meta` | Label, class id, source file names, stream properties |
| `/weld/alignment` | Sensor rate used, arc-on interval per stream, offset between streams |
| `/sensors/weld` | One message per sensor row |
| `/video/brightness` | Mean grey level per frame |
| `/video/frames` | Optional JPEG frames (`foxglove.CompressedImage`) |
| `/audio/rms` | Level per 50 ms window, when the weld has audio |

Time zero is the first video frame. Raw video and audio stay in their own
files and are referenced by name from `/weld/meta`.

## What the public sample files show

Checked against the 12 sample welds Intel publishes with its Manufacturing AI
Suite (one per class; two of them fetched with video):

- File stem: `<label>_<mm-dd-yy>-<run>-<class id>`. The class id is canonical;
  class 04 appears under two label spellings.
- Sensor columns: Pressure, CO2 Weld Flow, Feed, Primary Weld Current,
  Wire Consumed, Secondary Weld Voltage.
- No timestamp column and no travel speed.
- About 9 to 9.4 sensor rows per second of video.
- Video: 960x600, MPEG-4, 31 fps, 20 to 40 seconds.
- Arc-on durations from current and from video brightness agreed within 4%.
  Start times differed by 0.09 s on one weld and 1.21 s on the other.

## Assumptions to confirm on the full dataset

1. The full dataset pilot CSV contains Date and Time readings; the loader now
   preserves their measured intervals relative to the first sensor row.
   The camera-to-sensor clock offset remains unknown. Older public samples
   without timestamps still spread rows evenly over the video duration.
2. Audio shares the video's start time. Check `arc_on_audio_s` against
   `arc_on_video_s` in the manifest once audio is available.
3. Class id 03 is porosity. It is absent from the sample subset.

The offsets are recorded, not applied. Decide on a correction after looking
at their distribution across the full dataset.

The extracted pilot sample `04-01-23-0024-00` has 335 readings over 37.142 s,
and its FLAC has 608000 samples at 16 kHz (38 s). Its five JPG filenames
indicate April 3, while its sensor CSV says April 1; do not treat filename
timestamps as synchronized capture times without further evidence. The pilot
is a fillet weld, not a pipe-spool weld.

## Licence note

The Intel dataset is released for research use only. No dataset files are
included here.
