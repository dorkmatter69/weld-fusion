"""Write one weld to one MCAP file, and read a summary back.

Topics (all JSON, log time = nanoseconds since the first video frame):

    /weld/meta          one message: label, class, source files, stream info
    /weld/alignment     one message: sensor rate used and arc-on checks
    /sensors/weld       one message per sensor row
    /video/brightness   one message per video frame (mean grey level)
    /video/frames       optional JPEG frames (foxglove.CompressedImage)
    /audio/rms          one message per 50 ms window, when audio exists

Raw video and audio stay in their original files and are referenced from
/weld/meta, which keeps each MCAP small. Pass ``embed_every`` to also embed
every Nth frame so the weld can be scrubbed in Foxglove.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from mcap.reader import make_reader
from mcap.writer import Writer

from . import __version__
from .align import (
    audio_rms_envelope,
    build_alignment,
    scan_video,
    sensor_times_s,
)
from .dataset import WeldSample, load_sensor_table

GENERIC_SCHEMA = json.dumps({"type": "object"}).encode()
COMPRESSED_IMAGE_SCHEMA = json.dumps(
    {
        "title": "foxglove.CompressedImage",
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
            },
            "frame_id": {"type": "string"},
            "data": {"type": "string", "contentEncoding": "base64"},
            "format": {"type": "string"},
        },
    }
).encode()


def _ns(t_s: float) -> int:
    return int(round(t_s * 1e9))


def write_weld_mcap(
    sample: WeldSample,
    out_path: str | Path,
    sensor_rate_hz: float | None = None,
    embed_every: int = 0,
    jpeg_quality: int = 80,
) -> dict:
    """Convert one weld sample to MCAP. Returns the alignment as a dict."""
    if sample.video is None:
        raise ValueError(f"{sample.stem}: no video file, cannot set the timeline")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video = scan_video(sample.video)
    sensors = load_sensor_table(sample.sensors) if sample.sensors else None

    audio_rms = audio_times = None
    audio_info = None
    if sample.audio is not None:
        samples, rate = sf.read(str(sample.audio), always_2d=True)
        audio_rms, audio_times = audio_rms_envelope(samples, rate)
        audio_info = {
            "sample_rate_hz": int(rate),
            "channels": int(samples.shape[1]),
            "duration_s": samples.shape[0] / rate,
        }

    alignment = build_alignment(video, sensors, sensor_rate_hz, audio_rms, audio_times)

    with open(out_path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="", library=f"weldfusion {__version__}")
        generic = writer.register_schema("weldfusion.Json", "jsonschema", GENERIC_SCHEMA)

        def channel(topic: str, schema_id: int = generic) -> int:
            return writer.register_channel(
                topic=topic, message_encoding="json", schema_id=schema_id
            )

        def emit(ch: int, t_s: float, payload: dict, seq: int = 0) -> None:
            writer.add_message(
                channel_id=ch,
                log_time=_ns(t_s),
                publish_time=_ns(t_s),
                sequence=seq,
                data=json.dumps(payload).encode(),
            )

        meta = {
            "stem": sample.stem,
            "label": sample.label,
            "class_id": sample.class_id,
            "class_name": sample.class_name,
            "is_defect": sample.is_defect,
            "date": sample.date,
            "run": sample.run,
            "files": {
                "video": sample.video.name,
                "sensors": sample.sensors.name if sample.sensors else None,
                "audio": sample.audio.name if sample.audio else None,
            },
            "video": {
                "fps": video.fps,
                "frames": video.frame_count,
                "width": video.width,
                "height": video.height,
                "duration_s": video.duration_s,
            },
            "sensors": {
                "columns": sensors.columns if sensors else [],
                "rows": len(sensors) if sensors else 0,
                "dropped_columns": sensors.dropped_columns if sensors else [],
            },
            "audio": audio_info,
        }
        emit(channel("/weld/meta"), 0.0, meta)
        emit(channel("/weld/alignment"), 0.0, alignment.to_dict())

        if sensors is not None and len(sensors) and alignment.sensor_rate_hz > 0:
            ch = channel("/sensors/weld")
            times = sensor_times_s(len(sensors), alignment.sensor_rate_hz)
            for i, (t, row) in enumerate(zip(times, sensors.values)):
                emit(ch, float(t), dict(zip(sensors.columns, row.tolist())), seq=i)

        ch = channel("/video/brightness")
        for i, (t, b) in enumerate(zip(video.frame_times_s(), video.brightness)):
            emit(ch, float(t), {"frame": i, "mean_grey": float(b)}, seq=i)

        if embed_every and embed_every > 0:
            image_schema = writer.register_schema(
                "foxglove.CompressedImage", "jsonschema", COMPRESSED_IMAGE_SCHEMA
            )
            ch = channel("/video/frames", image_schema)
            cap = cv2.VideoCapture(str(sample.video))
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % embed_every == 0:
                    ok_enc, jpg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                    if ok_enc:
                        t = i / video.fps
                        ns = _ns(t)
                        emit(
                            ch,
                            t,
                            {
                                "timestamp": {"sec": ns // 10**9, "nsec": ns % 10**9},
                                "frame_id": "weld_camera",
                                "data": base64.b64encode(jpg.tobytes()).decode(),
                                "format": "jpeg",
                            },
                            seq=i,
                        )
                i += 1
            cap.release()

        if audio_rms is not None and audio_rms.size:
            ch = channel("/audio/rms")
            for i, (t, r) in enumerate(zip(audio_times, audio_rms)):
                emit(ch, float(t), {"rms": float(r)}, seq=i)

        writer.finish()

    return alignment.to_dict()


def summarize_mcap(path: str | Path) -> dict:
    """Per-topic message counts and time span, plus the meta and alignment."""
    topics: dict[str, dict] = {}
    meta = alignment = None
    with open(path, "rb") as f:
        for _schema, ch, msg in make_reader(f).iter_messages():
            t = topics.setdefault(
                ch.topic, {"count": 0, "first_s": None, "last_s": None}
            )
            t["count"] += 1
            ts = msg.log_time / 1e9
            t["first_s"] = ts if t["first_s"] is None else min(t["first_s"], ts)
            t["last_s"] = ts if t["last_s"] is None else max(t["last_s"], ts)
            if ch.topic == "/weld/meta":
                meta = json.loads(msg.data)
            elif ch.topic == "/weld/alignment":
                alignment = json.loads(msg.data)
    return {"topics": topics, "meta": meta, "alignment": alignment}


def read_topic(path: str | Path, topic: str) -> tuple[np.ndarray, list[dict]]:
    """All messages on one topic as (times in seconds, decoded payloads)."""
    times, payloads = [], []
    with open(path, "rb") as f:
        for _schema, _ch, msg in make_reader(f).iter_messages(topics=[topic]):
            times.append(msg.log_time / 1e9)
            payloads.append(json.loads(msg.data))
    return np.asarray(times), payloads
