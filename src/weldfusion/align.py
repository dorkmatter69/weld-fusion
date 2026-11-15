"""Put video, sensor and audio streams on one timeline and check the fit.

Time zero is the first video frame. The sensor CSVs in Intel's published
samples carry no timestamps, so sensor rows are spread uniformly over the
video duration (the same assumption Intel's own simulator makes) unless a
sensor rate is given. The arc-on interval is then detected independently in
each stream so the assumption can be checked instead of trusted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .dataset import SensorTable

CURRENT_COLUMN = "Primary Weld Current"
VOLTAGE_COLUMN = "Secondary Weld Voltage"


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    brightness: np.ndarray  # mean grey level per frame, shape (frame_count,)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def frame_times_s(self) -> np.ndarray:
        return np.arange(self.frame_count) / self.fps


def scan_video(path) -> VideoInfo:
    """Decode every frame once and record its mean brightness."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    brightness = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (96, 60), interpolation=cv2.INTER_AREA)
        brightness.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()
    return VideoInfo(
        fps=fps,
        frame_count=len(brightness),
        width=width,
        height=height,
        brightness=np.asarray(brightness, dtype=np.float64),
    )


def active_interval(signal: np.ndarray, times_s: np.ndarray, frac: float = 0.3):
    """First and last time a signal exceeds ``frac`` of its robust range.

    Returns ``None`` when the signal never rises clearly above its floor.
    """
    if signal.size < 4:
        return None
    lo, hi = np.percentile(signal, [5, 95])
    if hi - lo < 1e-9 or (hi - lo) < 0.05 * max(abs(hi), 1e-9):
        return None
    above = np.flatnonzero(signal > lo + frac * (hi - lo))
    if above.size == 0:
        return None
    return float(times_s[above[0]]), float(times_s[above[-1]])


@dataclass
class Alignment:
    video_duration_s: float
    sensor_rows: int
    sensor_rate_hz: float
    sensor_rate_source: str  # "uniform_over_video" or "given"
    arc_on_sensor_s: tuple[float, float] | None
    arc_on_video_s: tuple[float, float] | None
    arc_on_audio_s: tuple[float, float] | None
    start_offset_sensor_minus_video_s: float | None
    duration_ratio_sensor_over_video: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def sensor_times_s(rows: int, rate_hz: float) -> np.ndarray:
    return np.arange(rows) / rate_hz


def build_alignment(
    video: VideoInfo,
    sensors: SensorTable | None,
    sensor_rate_hz: float | None = None,
    audio_rms: np.ndarray | None = None,
    audio_rms_times_s: np.ndarray | None = None,
) -> Alignment:
    rows = len(sensors) if sensors is not None else 0
    if sensor_rate_hz:
        rate, source = float(sensor_rate_hz), "given"
    elif rows and video.duration_s > 0:
        rate, source = rows / video.duration_s, "uniform_over_video"
    else:
        rate, source = 0.0, "uniform_over_video"

    arc_sensor = None
    if sensors is not None and rows and rate > 0 and CURRENT_COLUMN in sensors.columns:
        arc_sensor = active_interval(
            sensors.column(CURRENT_COLUMN), sensor_times_s(rows, rate)
        )
    arc_video = active_interval(video.brightness, video.frame_times_s())
    arc_audio = None
    if audio_rms is not None and audio_rms_times_s is not None:
        arc_audio = active_interval(audio_rms, audio_rms_times_s)

    offset = ratio = None
    if arc_sensor and arc_video:
        offset = arc_sensor[0] - arc_video[0]
        video_len = arc_video[1] - arc_video[0]
        if video_len > 0:
            ratio = (arc_sensor[1] - arc_sensor[0]) / video_len

    return Alignment(
        video_duration_s=video.duration_s,
        sensor_rows=rows,
        sensor_rate_hz=rate,
        sensor_rate_source=source,
        arc_on_sensor_s=arc_sensor,
        arc_on_video_s=arc_video,
        arc_on_audio_s=arc_audio,
        start_offset_sensor_minus_video_s=offset,
        duration_ratio_sensor_over_video=ratio,
    )


def audio_rms_envelope(samples: np.ndarray, sample_rate: int, window_s: float = 0.05):
    """RMS level per window. Returns (rms, window start times in seconds)."""
    if samples.ndim > 1:
        samples = samples[:, 0]
    win = max(1, int(round(window_s * sample_rate)))
    n = samples.shape[0] // win
    if n == 0:
        return np.zeros(0), np.zeros(0)
    trimmed = samples[: n * win].astype(np.float64).reshape(n, win)
    rms = np.sqrt((trimmed**2).mean(axis=1))
    return rms, np.arange(n) * (win / sample_rate)
