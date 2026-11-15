# weldfusion

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

1. Sensor rows are spread evenly over the video duration. Intel's own
   simulator makes the same assumption. If the full CSVs include a Time
   column, use it instead; the loader currently drops Date and Time.
2. Audio shares the video's start time. Check `arc_on_audio_s` against
   `arc_on_video_s` in the manifest once audio is available.
3. Class id 03 is porosity. It is absent from the sample subset.

The offsets are recorded, not applied. Decide on a correction after looking
at their distribution across the full dataset.

## Licence note

The Intel dataset is released for research use only. No dataset files are
included here.
