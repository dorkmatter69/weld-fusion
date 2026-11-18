from weldfusion.sim_export import export_simulation


def test_export_runs_copied_mujoco_mock_plc():
    normal = export_simulation(seed=0)
    assert normal["source"]["generator"] == "MuJoCo WeldSim + Python MockPLC"
    assert normal["telemetry"][-1]["plc_state"] == "Done"
    assert normal["telemetry"][-1]["torch_x_m"] > 0.37
    stalled = export_simulation(seed=0, fault="stale_vision")
    assert stalled["telemetry"][-1]["plc_state"] == "Fault"
    assert stalled["telemetry"][-1]["torch_x_m"] < normal["telemetry"][-1]["torch_x_m"]
