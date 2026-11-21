import json

import numpy as np
import pytest

from weldfusion.dataset import SensorTable
from weldfusion.timestral_sensor_pilot import (
    arc_on_segment, baselines, interval_levels, main, prepare, run_pilot, score_quantiles)


def table(n=320, gaps=None, current=None):
    gaps = np.full(n - 1, 0.1) if gaps is None else np.asarray(gaps)
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    current = np.full(n, 100.0) if current is None else np.asarray(current)
    values = np.column_stack([current, 20 + 0.01 * np.arange(n)])
    return SensorTable(columns=["current", "voltage"], values=values,
                       dropped_columns=[], timestamps_s=times)


def test_future_rows_are_hidden_from_forecaster():
    w = prepare(table(), ["current", "voltage"], horizon=32, context=256)
    assert w.history.shape == (2, 256) and w.observed.shape == (2, 32)
    assert w.details["forecast_elapsed_s"][0] == pytest.approx(28.8)
    assert np.allclose(w.observed[1], 20 + 0.01 * np.arange(288, 320))
    with pytest.raises(ValueError, match="unavailable"):
        prepare(table(), ["force"])


def test_strict_rejects_jitter_rows_accepts_it():
    jitter = np.where(np.arange(319) % 2, 0.08, 0.14)
    with pytest.raises(ValueError, match="irregular"):
        prepare(table(gaps=jitter), ["current"], context=128, cadence="strict")
    w = prepare(table(gaps=jitter), ["current"], context=128, cadence="rows")
    assert w.history.shape == (1, 128) and w.step_s is None


def test_resample_uses_even_grid_and_real_held_out_times():
    jitter = np.where(np.arange(319) % 2, 0.08, 0.14)
    t = table(gaps=jitter)
    w = prepare(t, ["voltage"], context=128, horizon=32, cadence="resample")
    step = w.details["grid_step_s"]
    times = t.timestamps_s
    assert w.history.shape == (1, 128)
    assert np.allclose(w.observed[0], t.values[-32:, 1])          # real readings, not interpolated
    assert w.observed_offsets_s[0] == pytest.approx(times[-32] - times[-33])
    assert w.forecast_steps == int(np.ceil(w.observed_offsets_s[-1] / step - 1e-9))


def test_resample_refuses_missing_rows():
    gaps = np.full(319, 0.1)
    gaps[250] = 0.35
    with pytest.raises(ValueError, match="missing rows"):
        prepare(table(gaps=gaps), ["current"], context=64, cadence="resample")


def test_arc_on_window_excludes_arc_off_tail():
    current = np.r_[np.zeros(40), np.full(240, 180.0), np.zeros(40)]
    t = table(current=current)
    a, b = arc_on_segment(t, "current", 20, trim_s=1.0)
    assert (a, b) == (50, 270)
    w = prepare(t, ["current"], context=128, horizon=32, window="arc_on", arc_column="current")
    assert np.all(w.observed == 180) and w.details["segment_rows"] == [50, 270]


def test_scores_compare_quantiles_with_held_out_rows():
    w = prepare(table(), ["current"], horizon=32, context=256)
    levels = (0.1, 0.5, 0.9)
    q = np.stack([np.full((1, 32), 98.0), np.full((1, 32), 100.0), np.full((1, 32), 102.0)], axis=-1)
    result = score_quantiles(levels, q, w, ["current"])
    assert result["interval_quantiles"] == [0.1, 0.9]
    assert result["channels"]["current"] == {"mae": 0, "fraction_outside_interval": 0,
                                             "mean_interval_width": 4}
    assert baselines(w, ["current"])["persistence"]["channels"]["current"]["mae"] == 0
    with pytest.raises(ValueError, match="shape"):
        score_quantiles(levels, q[:, :10], w, ["current"])


def test_interval_levels_pick_nearest_knots():
    assert interval_levels((0.05, 0.1, 0.5, 0.9, 0.95)) == (0.1, 0.9)
    assert interval_levels((0.02, 0.5, 0.98), target=0.05) == (0.02, 0.98)


def test_run_pilot_reports_rejected_cadence_and_model_results():
    jitter = np.where(np.arange(319) % 2, 0.08, 0.14)

    def fake(history, steps):
        last = history[:, -1][:, None, None]
        return (0.1, 0.5, 0.9), np.repeat(last, steps, axis=1) + np.array([-1.0, 0.0, 1.0])

    runs = run_pilot(table(gaps=jitter), ["current"], {"timestral": fake}, context=128, horizon=32)
    by = {r["cadence"]: r for r in runs}
    assert "irregular" in by["strict"]["rejected"]
    assert by["rows"]["results"]["timestral"]["channels"]["current"]["mae"] == 0
    assert by["resample"]["results"]["timestral"]["channels"]["current"]["mae"] == pytest.approx(0)


def test_cli_dry_run_validates_local_csv_without_weights(tmp_path):
    csv = tmp_path / "synthetic.csv"
    csv.write_text("Date,Time,current,voltage\n" + "".join(
        f"04/01/2023,12:00:{i//10:02d}.{i%10}00000,100,20\n" for i in range(300)))
    output = tmp_path / "report.json"
    main([str(csv), str(output), "--columns", "current", "voltage", "--dry-run",
          "--window", "file_end"])
    report = json.loads(output.read_text())
    assert report["status"] == "validated_only" and report["source_rows"] == 300
    assert report["cross_stream_alignment_verified"] is False
    assert {r["cadence"] for r in report["runs"]} == {"strict", "rows", "resample"}
