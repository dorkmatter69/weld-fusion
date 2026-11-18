import numpy as np

from weld_twin import plc as P
from weld_twin.episode import run_episode


def test_closed_loop_beats_open_loop():
    for seed in range(3):
        cl = run_episode(seed, P.MockPLC())
        ol = run_episode(seed, P.MockPLC(), use_ml=False)
        assert cl.final_state == P.DONE
        assert cl.rms_mm < 1.0
        assert cl.rms_mm < ol.rms_mm / 2


def test_ml_stall_faults_and_stops_travel():
    r = run_episode(0, P.MockPLC(), stall_after_s=5.0)
    assert r.final_state == P.FAULT
    assert r.x_final < P.PlcConfig().x_end


def test_outlier_estimate_is_clamped():
    cfg = P.PlcConfig()
    r = run_episode(0, P.MockPLC(cfg), outlier_at_s=5.0)
    assert r.max_y_cmd_dev_mm <= cfg.max_offset * 1e3 + 1e-6


def test_nan_estimate_is_ignored():
    plc = P.MockPLC()
    plc.write_by_name("MAIN.bStart", True)
    plc.write_by_name("MAIN.fMlSeamY", float("nan"))
    plc.write_by_name("MAIN.nMlHeartbeat", 1)
    plc.tick(0.01)
    assert not plc.read_by_name("MAIN.bMlValid")


def test_mock_matches_pyads_surface():
    import pyads

    for name in ("read_by_name", "write_by_name", "open", "close"):
        assert hasattr(P.MockPLC, name) and hasattr(pyads.Connection, name)
