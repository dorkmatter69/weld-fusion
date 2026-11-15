import json

import numpy as np
import pytest

from weldfusion.align import active_interval
from weldfusion.cli import main
from weldfusion.dataset import discover_samples, load_sensor_table, parse_stem
from weldfusion.mcap_io import read_topic, summarize_mcap, write_weld_mcap
from weldfusion.synth import make_sample


def test_parse_stem_variants():
    a = parse_stem("good_weld_02-16-23-0081-00")
    assert a == {"label": "good_weld", "date": "02-16-23", "run": "0081", "class_id": 0}
    b = parse_stem("porosity_w-excessive_penetration_11-01-22-0161-04")
    assert b["class_id"] == 4 and b["label"] == "porosity_w-excessive_penetration"
    assert parse_stem("notes")["class_id"] is None


def test_discover_groups_files_by_stem(tmp_path):
    make_sample(tmp_path / "a", stem="good_weld_01-01-24-0001-00")
    make_sample(tmp_path / "b" / "deep", stem="spatter_01-02-24-0002-09", with_audio=False)
    samples = discover_samples(tmp_path)
    assert [s.stem for s in samples] == [
        "good_weld_01-01-24-0001-00",
        "spatter_01-02-24-0002-09",
    ]
    assert samples[0].modalities() == ["video", "sensors", "audio"]
    assert samples[1].modalities() == ["video", "sensors"]
    assert samples[0].is_defect is False and samples[1].is_defect is True
    assert samples[1].class_name == "spatter"


def test_sensor_table_drops_bookkeeping_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text(
        "Date,Time,Primary Weld Current ,Remarks ,Secondary Weld Voltage\n"
        "01/01/24,10:00:00,180.5,ok,24.1\n"
        "01/01/24,10:00:01,181.0,,24.3\n"
    )
    t = load_sensor_table(p)
    assert t.columns == ["Primary Weld Current", "Secondary Weld Voltage"]
    assert t.values.shape == (2, 2)
    assert set(t.dropped_columns) == {"Date", "Time", "Remarks"}


def test_active_interval_flat_signal_is_none():
    assert active_interval(np.ones(50), np.arange(50) / 10) is None


def test_convert_roundtrip_and_alignment(tmp_path):
    info = make_sample(tmp_path / "data", duration_s=6.0, arc_on_s=(1.5, 4.5))
    sample = discover_samples(tmp_path / "data")[0]
    out = tmp_path / "out" / "w.mcap"
    alignment = write_weld_mcap(sample, out, embed_every=30)

    summary = summarize_mcap(out)
    topics = summary["topics"]
    assert topics["/sensors/weld"]["count"] == info["rows"]
    assert topics["/video/brightness"]["count"] == info["frames"]
    assert topics["/video/frames"]["count"] == info["frames"] // 30
    assert topics["/audio/rms"]["count"] > 0
    assert summary["meta"]["class_id"] == 0
    assert summary["meta"]["sensors"]["columns"][3] == "Primary Weld Current"

    # Sensor rows span the video duration.
    assert alignment["sensor_rate_hz"] == pytest.approx(10.0, rel=0.02)
    assert topics["/sensors/weld"]["last_s"] == pytest.approx(5.9, abs=0.1)

    # Arc-on is found independently in all three streams and they agree.
    for key in ("arc_on_sensor_s", "arc_on_video_s", "arc_on_audio_s"):
        start, end = alignment[key]
        assert start == pytest.approx(1.5, abs=0.15)
        assert end == pytest.approx(4.5, abs=0.15)
    assert abs(alignment["start_offset_sensor_minus_video_s"]) < 0.15
    assert alignment["duration_ratio_sensor_over_video"] == pytest.approx(1.0, abs=0.08)

    # Values survive the round trip.
    times, rows = read_topic(out, "/sensors/weld")
    table = load_sensor_table(sample.sensors)
    assert rows[20]["Primary Weld Current"] == table.column("Primary Weld Current")[20]
    assert np.all(np.diff(times) > 0)


def test_detects_a_known_sensor_offset(tmp_path):
    # Sensor log starts 1 s late relative to video: arc appears 1 s earlier in it.
    info = make_sample(tmp_path / "d", duration_s=6.0, arc_on_s=(2.0, 5.0), with_audio=False)
    rows = info["sensors"].read_text().splitlines()
    info["sensors"].write_text("\n".join([rows[0]] + rows[11:]) + "\n")
    sample = discover_samples(tmp_path / "d")[0]
    alignment = write_weld_mcap(sample, tmp_path / "o.mcap", sensor_rate_hz=10.0)
    assert alignment["sensor_rate_source"] == "given"
    assert alignment["start_offset_sensor_minus_video_s"] == pytest.approx(-1.0, abs=0.15)


def test_cli_scan_convert_inspect(tmp_path, capsys):
    make_sample(tmp_path / "data", stem="good_weld_01-01-24-0001-00")
    make_sample(tmp_path / "data", stem="burnthrough_weld_01-03-24-0003-02", seed=1)
    assert main(["scan", str(tmp_path / "data")]) == 0
    assert "2 welds" in capsys.readouterr().out
    assert main(["convert", str(tmp_path / "data"), str(tmp_path / "out")]) == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert [m["class_name"] for m in manifest] == ["burnthrough", "good"]
    assert main(["inspect", str(tmp_path / "out" / manifest[0]["mcap"])]) == 0
    assert "/sensors/weld" in capsys.readouterr().out
