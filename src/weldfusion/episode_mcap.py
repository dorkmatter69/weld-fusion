"""Write any WeldEpisode v1 dict to MCAP (JSON messages), without video or audio deps.

Topics: /weld/meta (episode minus telemetry/events), /sensors/weld (one message per
telemetry row), /weld/events (one per event). Times are the episode's time_s.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcap.writer import Writer

from . import __version__

GENERIC_SCHEMA = json.dumps({"type": "object"}).encode()


def _ns(t_s: float) -> int:
    return int(round(float(t_s) * 1e9))


def write_episode_mcap(episode: dict, path: str | Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"/weld/meta": 1, "/sensors/weld": len(episode["telemetry"]),
              "/weld/events": len(episode.get("events", []))}
    with open(path, "wb") as handle:
        writer = Writer(handle)
        writer.start(profile="", library=f"weldfusion {__version__}")
        schema = writer.register_schema("weldfusion.Json", "jsonschema", GENERIC_SCHEMA)
        channels = {topic: writer.register_channel(topic=topic, message_encoding="json", schema_id=schema)
                    for topic in counts}

        def emit(topic, t_s, payload, seq):
            writer.add_message(channel_id=channels[topic], log_time=_ns(t_s), publish_time=_ns(t_s),
                               sequence=seq, data=json.dumps(payload).encode())

        meta = {k: v for k, v in episode.items() if k not in ("telemetry", "events")}
        emit("/weld/meta", 0.0, meta, 0)
        for seq, row in enumerate(episode["telemetry"]):
            emit("/sensors/weld", row["time_s"], row, seq)
        for seq, event in enumerate(episode.get("events", [])):
            emit("/weld/events", event["time_s"], event, seq)
        writer.finish()
    return counts
