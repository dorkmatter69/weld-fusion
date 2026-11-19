"""Checks derived from the observed Intel sample schema, without licensed data."""

import numpy as np
import pytest

from weldfusion.dataset import discover_samples, load_sensor_table


def test_intel_date_time_and_blank_trailing_remark(tmp_path):
    path = tmp_path / "04-01-23-0024-00.csv"
    path.write_text(
        "Date,Time,Part No,Pressure,CO2 Weld Flow,Feed,Primary Weld Current,"
        "Wire Consumed,Secondary Weld Voltage,Remarks \n"
        "4/1/2023,05:22:48.385,04-01-23-0024-00,4.38,-0.01,0.02,0.00,0.98,0.00\n"
        "4/1/2023,05:22:48.494,04-01-23-0024-00,4.38,0.01,0.02,0.00,1.15,0.00\n"
        "4/1/2023,05:22:48.603,04-01-23-0024-00,4.38,0.00,0.02,193.3,1.15,40.23\n"
    )
    table = load_sensor_table(path)
    assert np.allclose(table.timestamps_s, [0, 0.109, 0.218])
    assert "Remarks" in table.dropped_columns
    assert table.column("Primary Weld Current")[-1] == 193.3
    assert table.column("CO2 Weld Flow")[0] == -0.01


def test_non_monotonic_recorded_times_are_rejected(tmp_path):
    path = tmp_path / "backwards.csv"
    path.write_text(
        "Date,Time,Primary Weld Current\n"
        "4/1/2023,05:22:48.494,0\n"
        "4/1/2023,05:22:48.385,10\n"
    )
    with pytest.raises(ValueError, match="non-increasing"):
        load_sensor_table(path)


def test_manifest_joins_by_directory_and_is_not_a_sample(tmp_path):
    (tmp_path / "manifest.csv").write_text(
        "CATEGORY,SUBDIRS,WELD_TYPE,SPLIT\n"
        "Good,group-a/04-01-23-0024-00,FILLET,TRAIN\n"
        "Bad,group-b/04-01-23-0024-00,NON_FILLET,TEST\n"
    )
    for group in ("group-a", "group-b"):
        folder = tmp_path / group / "04-01-23-0024-00"
        folder.mkdir(parents=True)
        (folder / "04-01-23-0024-00.avi").write_bytes(b"avi")
    samples = discover_samples(tmp_path)
    assert len(samples) == 2
    assert {s.label for s in samples} == {"Good", "Bad"}
    assert {s.extra["manifest"]["SPLIT"] for s in samples} == {"TRAIN", "TEST"}
    assert sum(s.is_defect is False for s in samples) == 1
    assert len({s.output_name(duplicate_stem=True) for s in samples}) == 2
