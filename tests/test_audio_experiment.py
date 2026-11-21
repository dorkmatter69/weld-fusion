import numpy as np
import pytest

from weldfusion.audio_experiment import (
    chunk_starts, cohort_from_manifest, matched_n_fft, session_resplit, split_audit,
    stft_log_magnitude)


def fake_manifest():
    """Fabricated rows shaped like the Intel manifest; sessions span splits like the paper's."""
    rows = []
    def add(session, category, splits):
        for i, split in enumerate(splits):
            rows.append({"SUBDIRS": f"{session}/{session}-{i:02d}", "DIRECTORY": session,
                         "CATEGORY": category, "SPLIT": split})
    for g in range(10):
        add(f"good_{g}", "Good", ["TRAIN"] * 6 + ["VAL", "TEST"])
    for d in ("Porosity", "Spatter"):
        for k in range(4):
            add(f"{d}_{k}", d, ["VAL", "TEST"] * 3)
    return rows


def test_matched_window_keeps_paper_duration():
    assert matched_n_fft(16384) == 1365 or matched_n_fft(16384) == 1366
    assert abs(matched_n_fft(16384) / 16000 - 16384 / 192000) < 1e-4


def test_stft_shape_and_tone_peak():
    t = np.arange(16000) / 16000
    feats = stft_log_magnitude(np.sin(2 * np.pi * 1000 * t), 1366)
    assert feats.shape == ((16000 - 1366) // 683 + 1, 684)
    freqs = np.fft.rfftfreq(1366, 1 / 16000)
    assert abs(freqs[feats.mean(0).argmax()] - 1000) < 15


def test_paper_split_is_read_and_leakage_counted():
    cohort = cohort_from_manifest(fake_manifest())
    assert {r["split"] for r in cohort} == {"TRAIN", "VALIDATION", "TEST"}
    assert cohort[0]["flac"].endswith("good_0-00.flac")
    audit = split_audit(cohort)
    assert audit["sessions_spanning_splits"] == audit["sessions"]
    assert audit["test_good_welds_sharing_a_train_session"] == audit["test_good_welds"] == 10


def test_session_resplit_is_disjoint_and_complete():
    cohort = session_resplit(cohort_from_manifest(fake_manifest()), seed=3)
    audit = split_audit(cohort)
    assert audit["sessions_spanning_splits"] == 0
    assert audit["test_good_welds_sharing_a_train_session"] == 0
    for category, counts in audit["counts"].items():
        if category == "Good":
            assert all(counts[s] > 0 for s in ("TRAIN", "VALIDATION", "TEST"))
        else:
            assert counts["TRAIN"] == 0 and counts["VALIDATION"] > 0 and counts["TEST"] > 0
    assert session_resplit(cohort_from_manifest(fake_manifest()), seed=3) == cohort


def test_chunks_cover_every_frame():
    assert chunk_starts(11) == [0]
    assert chunk_starts(25) == [0, 11, 14]
    with pytest.raises(ValueError):
        chunk_starts(5)


def test_autoencoder_flags_unusual_audio():
    torch = pytest.importorskip("torch")
    from weldfusion.audio_experiment import run
    rng = np.random.default_rng(0)
    base = rng.normal(size=32)

    def weld(defect):
        f = base + 0.1 * rng.normal(size=(40, 32))
        if defect:
            f[:, 20:24] += 3.0
        return f.astype(np.float32)

    cohort = cohort_from_manifest(fake_manifest())
    features = {r["sample_id"]: weld(r["category"] != "Good") for r in cohort}
    torch.set_num_threads(2)
    import weldfusion.audio_experiment as ae
    original = ae.make_audio_ae
    ae.make_audio_ae = lambda torch, n, b, **kw: original(torch, n, b, width=32, **kw)
    try:
        report = run(cohort, features, torch, bottleneck=4, epochs=30, seed=1)
        raw = run(cohort, features, torch, bottleneck=4, epochs=30, seed=1, target="raw", batch=32)
    finally:
        ae.make_audio_ae = original
    assert report["test"]["overall_auc"] > 0.9 and raw["test"]["overall_auc"] > 0.9
    assert raw["target"] == "raw" and raw["batch"] == 32
    assert report["split_audit"]["sessions_spanning_splits"] > 0
    assert len(report["scores"]) == len(cohort)
