import numpy as np

from weldfusion.multimodal_benchmark import generate
from weldfusion.shift_fusion import (
    corrupt, features, forward, init_params, loss_and_grads, run_seed, train_gated)


def toy_blocks(n=40, seed=0):
    rng = np.random.default_rng(seed)
    B = {"e": rng.normal(size=(n, 3)), "A": rng.normal(size=(n, 3)), "Ra": rng.normal(size=(n, 3)),
         "S": rng.normal(size=(n, 6)), "Rs": rng.normal(size=(n, 3))}
    return B, (rng.random(n) < 0.4).astype(float)


def test_analytic_gradients_match_finite_differences():
    B, y = toy_blocks()
    p = init_params({k: B[k].shape[1] for k in B}, np.random.default_rng(1))
    for use_gate in (True, False):
        _, g = loss_and_grads(p, B, y, use_gate=use_gate)
        for key in p:
            for idx in list(np.ndindex(p[key].shape))[:4]:
                old = p[key][idx]
                p[key][idx] = old + 1e-6
                up = loss_and_grads(p, B, y, use_gate=use_gate)[0]
                p[key][idx] = old - 1e-6
                down = loss_and_grads(p, B, y, use_gate=use_gate)[0]
                p[key][idx] = old
                assert abs((up - down) / 2e-6 - g[key][idx]) < 1e-5, (key, idx, use_gate)


def test_gate_can_close_and_reduce_to_video_only():
    B, _ = toy_blocks()
    p = init_params({k: B[k].shape[1] for k in B}, np.random.default_rng(2))
    p["ga"][0] = p["gs"][0] = -60.0
    prob, h, cache = forward(p, B)
    assert np.allclose(h, B["e"], atol=1e-12) and cache["a"][3].max() < 1e-12


def test_corruptions_show_up_in_reliability_features():
    episode = generate()[0]
    rng = np.random.default_rng(0)
    clean, _ = features(episode)
    noisy, _ = features(corrupt(episode, "audio", "noisy", rng))
    missing, _ = features(corrupt(episode, "audio", "missing", rng))
    flagged, _ = features(corrupt(episode, "audio", "misaligned_flagged", rng))
    hidden, _ = features(corrupt(episode, "audio", "misaligned_hidden", rng))
    assert noisy["Ra"][:, 1].mean() > clean["Ra"][:, 1].mean() + 1        # spread rises
    assert missing["Ra"][:, 0].sum() <= clean["Ra"][:, 0].sum() - 7          # windows absent
    assert flagged["Ra"][0, 2] > clean["Ra"][0, 2] + 2                       # residual exposed
    assert hidden["Ra"][0, 2] == clean["Ra"][0, 2]                           # residual looks fine
    assert not np.allclose(hidden["A"], clean["A"])                          # but data moved
    assert np.array_equal(clean["Rs"], noisy["Rs"])                          # other modality untouched


def test_gated_model_learns_and_reports_every_condition():
    result = run_seed(42, steps=300)
    assert set(result["auc"]) >= {"clean", "audio_noisy", "audio_misaligned_hidden"}
    assert all(0 <= v <= 1 for c in result["auc"].values() for v in c.values())
    assert result["auc"]["clean"]["gated_shift"] > 0.8
    assert result["auc"]["clean"]["gated_shift"] > result["auc"]["clean"]["video_only"]
