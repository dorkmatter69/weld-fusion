"""Discover weld samples on disk and load their sensor tables.

A sample is identified by its file stem, e.g. ``good_weld_02-16-23-0081-00``:
``<label>_<mm-dd-yy>-<run>-<class id>``. The video (.avi), sensor table (.csv)
and audio (.flac/.wav) for one weld share that stem. Files are matched by stem
anywhere under the root, so the loader does not depend on the folder layout.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

STEM_RE = re.compile(
    r"^(?P<label>.+?)_(?P<date>\d{2}-\d{2}-\d{2})-(?P<run>\d{4})-(?P<cls>\d{2})$"
)

# Class ids as they appear in Intel's published sample files. The id in the
# file name is canonical: two label spellings exist for class 04.
CLASS_NAMES = {
    0: "good",
    1: "excessive_penetration",
    2: "burnthrough",
    3: "porosity",  # inferred: not in the public sample subset; confirm on full data
    4: "porosity_w_excessive_penetration",
    5: "undercut",
    6: "overlap",
    7: "lack_of_fusion",
    8: "excessive_convexity",
    9: "spatter",
    10: "warping",
    11: "crater_cracks",
}

VIDEO_EXT = {".avi", ".mp4"}
AUDIO_EXT = {".flac", ".wav"}
# Non-numeric bookkeeping columns seen in Intel's own tooling.
NON_SIGNAL_COLUMNS = {"date", "time", "remarks", "part no"}


def _parse_csv_clock(date: str, time: str) -> datetime:
    for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(f"{date} {time}", fmt)
        except ValueError:
            pass
    raise ValueError(f"invalid sensor date/time: {date} {time}")


@dataclass
class WeldSample:
    stem: str
    label: str
    class_id: int | None
    date: str | None
    run: str | None
    video: Path | None = None
    sensors: Path | None = None
    audio: Path | None = None
    extra: dict = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        if self.class_id is None:
            return self.label
        return CLASS_NAMES.get(self.class_id, self.label)

    @property
    def is_defect(self) -> bool | None:
        if self.class_id is not None:
            return self.class_id != 0
        return False if self.label.lower() == "good" else None

    def modalities(self) -> list[str]:
        return [
            name
            for name, path in (
                ("video", self.video),
                ("sensors", self.sensors),
                ("audio", self.audio),
            )
            if path is not None
        ]

    def output_name(self, duplicate_stem: bool = False) -> str:
        """Stable basename; qualify repeated stems by their source subdirectory."""
        if not duplicate_stem:
            return self.stem
        source = self.extra.get("relative_subdir")
        if source is None:
            raise ValueError(f"no source subdirectory for duplicated sample {self.stem}")
        digest = hashlib.sha256(source.encode()).hexdigest()[:12]
        return f"{self.stem}-{digest}"


def parse_stem(stem: str) -> dict:
    m = STEM_RE.match(stem)
    if not m:
        return {"label": stem, "date": None, "run": None, "class_id": None}
    return {
        "label": m["label"],
        "date": m["date"],
        "run": m["run"],
        "class_id": int(m["cls"]),
    }


def discover_samples(root: str | Path) -> list[WeldSample]:
    """Find every weld under ``root`` and group its files by stem."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    manifest_rows = {}
    manifest = root / "manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                key = row["SUBDIRS"].strip("/")
                if key in manifest_rows:
                    raise ValueError(f"duplicate manifest subdirectory: {key}")
                manifest_rows[key] = row
    samples: dict[tuple[Path, str], WeldSample] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path == manifest:
            continue
        ext = path.suffix.lower()
        if ext not in VIDEO_EXT | AUDIO_EXT | {".csv"}:
            continue
        stem = path.stem
        key = (path.parent, stem)
        if key not in samples:
            sample = WeldSample(stem=stem, **parse_stem(stem))
            relative_subdir = path.parent.relative_to(root).as_posix()
            sample.extra["relative_subdir"] = relative_subdir
            record = manifest_rows.get(relative_subdir)
            if record:
                sample.label = record["CATEGORY"]
                sample.class_id = 0 if sample.label.lower() == "good" else None
                sample.extra["manifest"] = record
            samples[key] = sample
        sample = samples[key]
        if ext in VIDEO_EXT:
            sample.video = path
        elif ext in AUDIO_EXT:
            sample.audio = path
        else:
            sample.sensors = path
    return sorted(samples.values(), key=lambda s: (s.stem, str(s.video or s.sensors or s.audio)))


@dataclass
class SensorTable:
    columns: list[str]
    values: np.ndarray  # shape (rows, len(columns)), float64
    dropped_columns: list[str]
    timestamps_s: np.ndarray | None = None  # relative to the first CSV reading
    first_timestamp_local: str | None = None  # original string; timezone unknown

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.columns.index(name)]

    def __len__(self) -> int:
        return self.values.shape[0]


def load_sensor_table(path: str | Path) -> SensorTable:
    """Read a sensor CSV, keeping numeric signal columns only.

    Header names are stripped of whitespace. Bookkeeping columns (Date, Time,
    Remarks, Part No) and any column that is not fully numeric are dropped and
    reported in ``dropped_columns``.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    keep_idx, keep_names, dropped = [], [], []
    for i, name in enumerate(header):
        if name.lower() in NON_SIGNAL_COLUMNS or not name:
            dropped.append(name)
            continue
        try:
            [float(r[i]) for r in rows]
        except (ValueError, IndexError):
            dropped.append(name)
            continue
        keep_idx.append(i)
        keep_names.append(name)
    values = np.array([[float(r[i]) for i in keep_idx] for r in rows], dtype=np.float64)
    if values.size == 0:
        values = values.reshape(0, len(keep_idx))
    timestamps = None
    first_timestamp_local = None
    if "Date" in header and "Time" in header and rows:
        date_i, time_i = header.index("Date"), header.index("Time")
        first_timestamp_local = f"{rows[0][date_i]} {rows[0][time_i]}"
        try:
            clock = [_parse_csv_clock(r[date_i], r[time_i]) for r in rows]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid Date/Time reading in {path}") from exc
        timestamps = np.asarray(
            [(instant - clock[0]).total_seconds() for instant in clock],
            dtype=np.float64,
        )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"non-increasing Date/Time readings in {path}")
    return SensorTable(
        columns=keep_names,
        values=values,
        dropped_columns=dropped,
        timestamps_s=timestamps,
        first_timestamp_local=first_timestamp_local,
    )
