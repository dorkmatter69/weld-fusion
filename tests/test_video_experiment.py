import json

import numpy as np
import pytest

from weldfusion.video_experiment import load_index, temporal_contexts, peak_two_seconds


def test_temporal_context_never_sees_overlapping_source_windows():
    features = np.arange(196, dtype=np.float32).reshape(-1, 1)
    contexts = list(temporal_contexts(features))
    covered = sorted(int(i) for indices, _ in contexts for i in indices)
    assert covered == list(range(196))
    assert all(np.all(np.diff(indices) >= 64) for indices, _ in contexts)


def test_rejects_session_leakage(tmp_path):
    rows = [
        {"sample_id": "a", "session_id": "one", "split": "TRAIN", "category": "Good", "feature_path": "a.npz"},
        {"sample_id": "b", "session_id": "one", "split": "VALIDATION", "category": "Good", "feature_path": "b.npz"},
        {"sample_id": "c", "session_id": "two", "split": "VALIDATION", "category": "Overlap", "feature_path": "c.npz"},
        {"sample_id": "d", "session_id": "three", "split": "TEST", "category": "Good", "feature_path": "d.npz"},
        {"sample_id": "e", "session_id": "three", "split": "TEST", "category": "Overlap", "feature_path": "e.npz"},
    ]
    path = tmp_path / "index.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="session spans splits"):
        load_index(path)


def test_two_second_peak_uses_real_feature_timing():
    centers = np.arange(100) / 25.0
    errors = np.zeros(100)
    errors[20:70] = 2
    assert peak_two_seconds(errors, centers) == 2
