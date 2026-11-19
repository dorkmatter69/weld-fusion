import numpy as np
import pytest

from weldfusion.multimodal_benchmark import fit_clock, generate, run, windows_for


def test_clock_fit_uses_independent_landmark_and_rejects_bad_fit():
    reference = np.array([2, 6, 14, 18.0])
    local = 0.4 + 1.003 * reference
    slope, intercept, residual = fit_clock(reference, local)
    assert np.max(abs(slope * local + intercept - reference)) < 1e-10
    assert residual < 1e-10
    with pytest.raises(ValueError, match="independent"):
        fit_clock(reference, local + [0, 0, 0, 0.4])


def test_sessions_are_disjoint_and_alignment_changes_binning():
    episodes = generate()
    assert len(episodes) == 96
    seen = {}
    for episode in episodes:
        seen.setdefault(episode["session_id"], episode["split"])
        assert seen[episode["session_id"]] == episode["split"]
        x, labels = windows_for(episode)
        assert x.shape == (20, 7) and labels.shape == (20,)
    assert any(not np.array_equal(windows_for(e)[0], windows_for(e, False)[0]) for e in episodes)


def test_fixed_seed_report_tracks_all_models_and_frozen_test_split():
    report = run()
    assert report["data_kind"] == "simulation"
    assert report["session_count"] == 12
    assert report["splits"] == {"TRAIN": 64, "VALIDATION": 16, "TEST": 16}
    assert len(report["test_windows"]) == 320
    assert set(report["per_category"]) == {"Overlap", "Gas loss", "Spatter"}
    assert all(0 <= m["auc"] <= 1 for m in report["results"].values())
    assert report["results"]["continuous"]["auc"] > report["results"]["late_video"]["auc"]
