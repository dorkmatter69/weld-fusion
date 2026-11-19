"""Build a local, reviewable EmbeddingGemma 2 index from short weld clips.

No network upload: weights are downloaded by sentence-transformers on first
use, while private media stays on the machine running this script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def validate_manifest(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must be a nonempty JSON array")
    seen = set()
    for row in rows:
        for key in ("sample_id", "session_id", "split", "category", "video_path", "audio_path"):
            if not row.get(key):
                raise ValueError(f"missing {key}")
        if row["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(row["sample_id"])


def windows(duration, width, hop):
    if duration <= 0 or width <= 0 or hop <= 0:
        raise ValueError("duration, width and hop must be positive")
    start = 0.0
    while start < duration:
        end = min(duration, start + width)
        if end - start < min(1.0, width):
            break
        yield round(start, 6), round(end, 6)
        start += hop


def ffprobe_seconds(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def encode_clips(manifest, output, width=4.0, hop=2.0, fps=4):
    from sentence_transformers import SentenceTransformer

    rows = json.loads(manifest.read_text())
    validate_manifest(rows)
    model = SentenceTransformer("google/embeddinggemma-2")
    results = []
    for row in rows:
        video = Path(row["video_path"])
        audio = Path(row["audio_path"])
        if not video.is_file() or not audio.is_file():
            raise FileNotFoundError(f"missing local media for {row['sample_id']}")
        duration = min(ffprobe_seconds(video), ffprobe_seconds(audio))
        with tempfile.TemporaryDirectory(prefix="weld-embed-") as temp:
            for i, (start, end) in enumerate(windows(duration, width, hop)):
                mp4 = Path(temp) / "clip.mp4"
                wav = Path(temp) / "clip.wav"
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(start),
                                "-i", str(video), "-t", str(end-start), "-vf", f"fps={fps}",
                                "-an", "-c:v", "libx264", "-preset", "ultrafast", str(mp4)], check=True)
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(start),
                                "-i", str(audio), "-t", str(end-start), "-ac", "1", "-ar", "16000",
                                str(wav)], check=True)
                # Separate embeddings permit within-modality nearest-neighbor search.
                # Shared coordinates do not establish synchronized device clocks.
                vectors = {}
                for mode, key in (("video", "video"), ("audio", "audio")):
                    encoded = model.encode({key: str(mp4 if mode == "video" else wav)},
                                           normalize_embeddings=True)
                    vector = np.asarray(encoded, dtype=np.float32).reshape(-1)
                    if vector.shape != (768,) or not np.isfinite(vector).all():
                        raise ValueError("unexpected embedding dimension or nonfinite value")
                    vectors[mode] = vector.tolist()
                results.append({"id": f"{row['sample_id']}:{i}", "sample_id": row["sample_id"],
                                "session_id": row["session_id"], "split": row["split"],
                                "category": row["category"], "start_s": start, "end_s": end,
                                "embeddings": vectors})
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite existing index")
    output.write_text(json.dumps({"schema_version": 1, "model": "google/embeddinggemma-2",
                                  "clip_seconds": width, "hop_seconds": hop, "video_fps": fps,
                                  "clock_alignment_verified": False, "segments": results}) + "\n")
    return len(results)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--clip-seconds", type=float, default=4.0)
    p.add_argument("--hop-seconds", type=float, default=2.0)
    p.add_argument("--video-fps", type=int, default=4)
    args = p.parse_args(argv)
    if args.video_fps < 1 or args.video_fps * args.clip_seconds > 58:
        p.error("clip consumes too many video frames for default token budget")
    print(f"Indexed {encode_clips(args.manifest, args.output, args.clip_seconds, args.hop_seconds, args.video_fps)} segments")


if __name__ == "__main__":
    main()
