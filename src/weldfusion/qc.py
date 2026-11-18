"""Per-weld screening. All event times stay on their original stream clocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from .align import active_interval, scan_video
from .dataset import load_sensor_table


def _interval(values, times):
    result = active_interval(values, times)
    return None if result is None else {"start_s": result[0], "end_s": result[1],
                                         "method": "5_95_percentile_30pct_screening_v1"}


def check_weld(sample_dir: str | Path) -> dict:
    sample_dir = Path(sample_dir)
    stem = sample_dir.name
    result = {"sample_id": stem, "status": "unresolved", "clock_alignment_verified": False,
              "streams": {}, "event_candidates": {}, "warnings": []}
    csv_path = sample_dir / f"{stem}.csv"
    avi_path = sample_dir / f"{stem}.avi"
    flac_path = sample_dir / f"{stem}.flac"
    if csv_path.is_file():
        try:
            table = load_sensor_table(csv_path)
            t = table.timestamps_s
            result["streams"]["sensors"] = {
                "rows": len(table), "first_timestamp_local": table.first_timestamp_local,
                "span_s": float(t[-1] - t[0]) if t is not None and len(t) > 1 else None,
                "interval_median_s": float(np.median(np.diff(t))) if t is not None and len(t) > 1 else None,
                "columns": table.columns, "dropped_columns": table.dropped_columns,
            }
            if t is None:
                result["warnings"].append("sensor timestamps unavailable")
            elif "Primary Weld Current" in table.columns:
                result["event_candidates"]["sensor_current"] = _interval(
                    table.column("Primary Weld Current"), t)
            else:
                result["warnings"].append("current column unavailable")
        except (OSError, ValueError, IndexError) as exc:
            result["warnings"].append(f"sensor read failed: {exc}")
    else:
        result["warnings"].append("sensor file missing")
    if avi_path.is_file():
        try:
            video = scan_video(avi_path)
            if not video.frame_count:
                raise ValueError("zero decoded frames")
            result["streams"]["video"] = {"frames": video.frame_count, "fps": video.fps,
                                             "duration_s": video.duration_s,
                                             "width": video.width, "height": video.height}
            result["event_candidates"]["video_brightness"] = _interval(
                video.brightness, video.frame_times_s())
        except (OSError, ValueError, cv2.error) as exc:
            result["warnings"].append(f"video decode failed: {exc}")
    else:
        result["warnings"].append("video file missing")
    if flac_path.is_file():
        try:
            # Read bounded blocks, so a long recording does not occupy unbounded RAM.
            with sf.SoundFile(flac_path) as audio:
                sample_rate, frames, channels = audio.samplerate, len(audio), audio.channels
                rms, times = [], []
                block = max(1, round(sample_rate * 0.05))
                while True:
                    chunk = audio.read(block, dtype="float32", always_2d=True)
                    if len(chunk) == 0:
                        break
                    times.append((len(times) * block) / sample_rate)
                    rms.append(float(np.sqrt(np.mean(np.square(chunk[:, 0].astype(np.float64))))))
            result["streams"]["audio"] = {"samples": frames, "sample_rate_hz": sample_rate,
                                             "channels": channels, "duration_s": frames / sample_rate}
            result["event_candidates"]["audio_rms"] = _interval(np.asarray(rms), np.asarray(times))
        except (OSError, ValueError, RuntimeError) as exc:
            result["warnings"].append(f"audio decode failed: {exc}")
    else:
        result["warnings"].append("audio file missing")
    events = result["event_candidates"]
    if events.get("sensor_current") and events.get("video_brightness"):
        result["candidate_differences_s"] = {
            "sensor_minus_video_start": events["sensor_current"]["start_s"] - events["video_brightness"]["start_s"],
            "sensor_minus_video_end": events["sensor_current"]["end_s"] - events["video_brightness"]["end_s"],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    result = check_weld(args.sample_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output} ({result['status']}; {len(result['warnings'])} warnings)")
    return 0 if len(result["streams"]) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
