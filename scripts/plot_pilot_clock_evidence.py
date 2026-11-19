"""Plot MCAP pilot signals on their original, unaligned file time bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcap.reader import make_reader


CHANNELS = (
    ("/sensors/weld", "Primary Weld Current", "Sensor current (A)", "#1565c0", "CSV first row"),
    ("/video/brightness", "mean_grey", "Video mean brightness", "#ef6c00", "AVI first frame"),
    ("/audio/rms", "rms", "Audio RMS", "#2e7d32", "FLAC first sample"),
)


def render(source: Path, destination: Path) -> None:
    traces = {topic: ([], []) for topic, *_ in CHANNELS}
    boundaries = {}
    with source.open("rb") as f:
        for _, channel, msg in make_reader(f).iter_messages():
            if channel.topic == "/weld/alignment":
                boundaries = json.loads(msg.data)
            if channel.topic in traces:
                times, values = traces[channel.topic]
                field = next(field for topic, field, *_ in CHANNELS if topic == channel.topic)
                times.append(msg.log_time / 1e9)
                values.append(float(json.loads(msg.data)[field]))
    if not boundaries or any(not times for times, _ in traces.values()):
        raise ValueError("MCAP lacks alignment metadata or a required trace")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    keys = ("arc_on_sensor_s", "arc_on_video_s", "arc_on_audio_s")
    for ax, (topic, _field, title, color, origin), key in zip(axes, CHANNELS, keys):
        times, values = traces[topic]
        start, end = boundaries[key]
        ax.plot(times, values, color=color, linewidth=1.1)
        ax.axvline(start, color="black", linestyle="--", alpha=.7)
        ax.axvline(end, color="black", linestyle="--", alpha=.7)
        ax.text(start, max(values) * .95, f"on {start:.3f}s", rotation=90, va="top", ha="right")
        ax.text(end, max(values) * .95, f"off {end:.3f}s", rotation=90, va="top", ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Time zero: {origin}", loc="left", fontsize=9)
        ax.grid(alpha=.22)
    axes[-1].set_xlabel("Seconds since each stream's own start (not synchronized)")
    axes[-1].set_xlim(0, 38)
    fig.suptitle("Intel weld 04-01-23-0024-00: exploratory arc boundaries\n"
                 "Dashed lines are screening thresholds; no shared-clock mapping applied", fontsize=13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, metadata={"Description": "Unaligned file-relative welding stream evidence"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.mcap, args.output)
