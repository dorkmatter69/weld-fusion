from weldfusion.episode_export import export_pilot


def test_recorded_episode_keeps_measured_times_and_unknown_alignment(tmp_path):
    sample = tmp_path / "04-01-23-0024-00"
    sample.mkdir()
    (sample / "images").mkdir()
    (sample / "images" / "20230403_134614_025.jpg").write_bytes(b"jpeg")
    (sample / f"{sample.name}.avi").write_bytes(b"avi")
    (sample / f"{sample.name}.flac").write_bytes(b"flac")
    (sample / f"{sample.name}.csv").write_text(
        "Date,Time,Part No,Pressure,CO2 Weld Flow,Feed,Primary Weld Current,"
        "Wire Consumed,Secondary Weld Voltage,Remarks \n"
        "4/1/2023,05:22:48.385,04-01-23-0024-00,4.38,-0.01,0.02,0.00,0.98,0.00\n"
        "4/1/2023,05:22:48.494,04-01-23-0024-00,4.38,0.01,0.02,193.3,1.15,40.23\n"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "CATEGORY,WELD_TYPE,THICKNESS_MM,STEEL_TYPE,SAMPLES,CURRENT_A,VOLTAGE_V,"
        "GAS_BAR,ROBOT_SPEED_CPM,DIRECTORY,SUBDIRS,SPLIT\n"
        "Good,FILLET,7,FE410,19,165,21.0,4.5,25,group,group/04-01-23-0024-00,TRAIN\n"
    )
    episode = export_pilot(sample, "gs://private/pilot/sample", manifest)
    assert episode["source"]["kind"] == "recorded"
    assert episode["process"]["weld_type"] == "FILLET"
    assert episode["labels"]["category"] == "Good"
    assert episode["timeline"]["clock_alignment_verified"] is False
    assert [r["time_s"] for r in episode["telemetry"]] == [0, 0.109]
    assert episode["telemetry"][0]["co2_flow_raw"] == -0.01
    assert episode["streams"]["video"] == "gs://private/pilot/sample/04-01-23-0024-00.avi"
