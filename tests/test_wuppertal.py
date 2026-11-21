import numpy as np

from weldfusion.episode_mcap import write_episode_mcap
from weldfusion.wuppertal import (
    FEATURE_NAMES, label_stretches, pulse_features, segment_episode, segment_ids)


def pulses(n, bg=45.0, peak=500.0, v_dip=False):
    t = np.arange(200)
    shape = np.exp(-0.5 * ((t - 35) / 12.0) ** 2)
    I = np.tile(bg + (peak - bg) * shape, (n, 1))
    V = np.tile(18 + 11 * shape, (n, 1))
    if v_dip:
        V[:, 80:84] = 4.0
    return V, I


def test_pulse_features_capture_peak_background_and_dips():
    V, I = pulses(3, v_dip=True)
    X = pulse_features(V, I)
    f = dict(zip(FEATURE_NAMES, X[0]))
    assert X.shape == (3, len(FEATURE_NAMES))
    assert abs(f["i_peak"] - 500) < 1e-3 and abs(f["i_background"] - 45) < 1
    assert f["peak_index"] == 35 and f["short_circuit_samples"] == 4
    assert f["v_min_after_peak"] == 4.0


def test_segments_split_non_contiguous_runs_without_joining():
    runs = np.array([7, 7, 8, 7, 7, 8])
    assert list(segment_ids(runs)) == ["7/0", "7/0", "8/0", "7/1", "7/1", "8/1"]


def test_label_stretches():
    assert label_stretches([-1, -1, 0, 0, 0, 1, -1]) == [(0, 2, -1), (2, 5, 0), (5, 6, 1), (6, 7, -1)]


def test_segment_episode_averages_to_intel_timescale_and_marks_assumptions(tmp_path):
    V, I = pulses(130)
    labels = np.r_[np.full(60, 0), np.full(40, -1), np.full(30, 1)]
    episode = segment_episode("8", "1", 2, V, I, labels)
    assert episode["schema_version"] == "1.0.0"
    assert episode["timeline"]["pulse_period_assumed"] is True
    assert episode["timeline"]["pulses_per_window"] == 58
    assert len(episode["telemetry"]) == 2                         # 130 // 58 windows; tail dropped
    row = episode["telemetry"][1]
    assert row["time_s"] == 0.116 and abs(row["current_a"] - I[0].mean()) < 1e-6
    assert episode["events"][0] == {"time_s": 0.0, "end_s": 0.12, "type": "quality_label",
                                    "label": "bad", "pulses": 60}
    assert episode["labels"]["good_pulses"] == 16 and episode["source"]["licence"] == "CC-BY-4.0"
    counts = write_episode_mcap(episode, tmp_path / "e.mcap")
    assert counts["/sensors/weld"] == 2 and (tmp_path / "e.mcap").stat().st_size > 0
