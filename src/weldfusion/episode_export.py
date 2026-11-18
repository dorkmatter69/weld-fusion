"""Export an extracted Intel sample to WeldEpisode JSON without invented clocks.

Usage: python -m weldfusion.episode_export SAMPLE_DIR OUTPUT.json
       --manifest /path/to/manifest.csv --gcs-prefix gs://bucket/pilot/sample-id
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .dataset import load_sensor_table


def _manifest_record(path: Path | None, sample_id: str) -> dict:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        matches = [
            row for row in csv.DictReader(file)
            if row["SUBDIRS"].rstrip("/").split("/")[-1] == sample_id
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest record for {sample_id}, got {len(matches)}")
    return matches[0]


def export_pilot(sample_dir: str | Path, gcs_prefix: str | None = None,
                 manifest: str | Path | None = None) -> dict:
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name
    csv_path = sample_dir / f"{sample_id}.csv"
    video = sample_dir / f"{sample_id}.avi"
    audio = sample_dir / f"{sample_id}.flac"
    images = sorted((sample_dir / "images").glob("*.jpg"))
    for path in (csv_path, video, audio):
        if not path.is_file():
            raise FileNotFoundError(path)
    sensors = load_sensor_table(csv_path)
    if sensors.timestamps_s is None:
        raise ValueError("pilot requires measured CSV Date and Time columns")
    record = _manifest_record(Path(manifest) if manifest else None, sample_id)
    source_identity = record.get("SUBDIRS") or (gcs_prefix or str(sample_dir.resolve()))
    identity_hash = hashlib.sha256(source_identity.encode()).hexdigest()[:12]

    def ref(path: Path) -> str:
        relative = path.relative_to(sample_dir).as_posix()
        return f"{gcs_prefix.rstrip('/')}/{relative}" if gcs_prefix else str(path)

    columns = {
        "Pressure": "pressure_raw", "CO2 Weld Flow": "co2_flow_raw",
        "Feed": "feed_raw", "Primary Weld Current": "current_a",
        "Wire Consumed": "wire_consumed_raw",
        "Secondary Weld Voltage": "voltage_v",
    }
    telemetry = []
    for t, row in zip(sensors.timestamps_s, sensors.values):
        reading = {"time_s": float(t)}
        reading.update({columns.get(k, k): float(value)
                        for k, value in zip(sensors.columns, row)})
        telemetry.append(reading)

    def number(name: str):
        value = record.get(name)
        return float(value) if value else None

    return {
        "schema_version": "1.0.0", "episode_id": f"intel-{sample_id}-{identity_hash}",
        "source": {"kind": "recorded", "dataset": "Intel Robotic Welding Multimodal Dataset",
                   "archive_subdir": record.get("SUBDIRS"), "use": "research"},
        "process": {"weld_type": record.get("WELD_TYPE"),
                    "steel_type": record.get("STEEL_TYPE"),
                    "thickness_mm": number("THICKNESS_MM")},
        "recipe": {"current_a": number("CURRENT_A"), "voltage_v": number("VOLTAGE_V"),
                   "gas_bar": number("GAS_BAR"), "robot_speed_cpm": number("ROBOT_SPEED_CPM")},
        "labels": {"category": record.get("CATEGORY"), "split": record.get("SPLIT")},
        "timeline": {"time_basis": "sensor_first_row_elapsed_s",
                     "clock_alignment_verified": False,
                     "first_sensor_timestamp_local": sensors.first_timestamp_local,
                     "audio_video_sensor_offset_s": None,
                     "image_capture_time_verified": False},
        "streams": {"sensors": ref(csv_path), "video": ref(video), "audio": ref(audio),
                    "post_weld_images": [ref(p) for p in images]},
        "telemetry": telemetry, "events": [], "reviews": [],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gcs-prefix")
    args = parser.parse_args(argv)
    episode = export_pilot(args.sample_dir, args.gcs_prefix, args.manifest)
    args.output.write_text(json.dumps(episode, indent=2))
    print(f"Wrote {len(episode['telemetry'])} readings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
