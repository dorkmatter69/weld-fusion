"""Generate small synthetic welds in the dataset's file layout.

Used by the tests and for trying the pipeline before real data is available.
Each sample has a dark lead-in, a bright arc-on interval and a dark tail, with
current, voltage and audio level rising and falling at the same moments.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

COLUMNS = [
    "Pressure",
    "CO2 Weld Flow",
    "Feed",
    "Primary Weld Current",
    "Wire Consumed",
    "Secondary Weld Voltage",
]


def make_sample(
    out_dir: str | Path,
    stem: str = "good_weld_01-01-24-0001-00",
    duration_s: float = 6.0,
    arc_on_s: tuple[float, float] = (1.5, 4.5),
    fps: int = 30,
    sensor_hz: float = 10.0,
    audio_hz: int = 16000,
    with_audio: bool = True,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    on, off = arc_on_s

    video_path = out_dir / f"{stem}.avi"
    w, h = 160, 100
    vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    n_frames = int(round(duration_s * fps))
    for i in range(n_frames):
        t = i / fps
        level = 200 if on <= t <= off else 20
        frame = np.clip(rng.normal(level, 5, (h, w, 3)), 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()

    csv_path = out_dir / f"{stem}.csv"
    n_rows = int(round(duration_s * sensor_hz))
    wire = 0.0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for i in range(n_rows):
            t = i / sensor_hz
            arc = on <= t <= off
            wire += 5.0 if arc else 0.0
            writer.writerow(
                [
                    round(1.5 + rng.normal(0, 0.01), 2),
                    round((12.0 if arc else 0.0) + rng.normal(0, 0.05), 2),
                    round(48.0 if arc else 0.0, 2),
                    round((180.0 if arc else 0.0) + (rng.normal(0, 3) if arc else 0), 2),
                    round(wire, 2),
                    round((24.0 if arc else 0.0) + (rng.normal(0, 0.4) if arc else 0), 2),
                ]
            )

    audio_path = None
    if with_audio:
        audio_path = out_dir / f"{stem}.flac"
        t = np.arange(int(duration_s * audio_hz)) / audio_hz
        gain = np.where((t >= on) & (t <= off), 0.4, 0.005)
        sf.write(str(audio_path), (rng.normal(0, 1, t.shape) * gain).astype(np.float32), audio_hz)

    return {
        "stem": stem,
        "video": video_path,
        "sensors": csv_path,
        "audio": audio_path,
        "duration_s": duration_s,
        "arc_on_s": arc_on_s,
        "frames": n_frames,
        "rows": n_rows,
    }
