from types import SimpleNamespace

import numpy as np
import pytest

from weldfusion.dataset import SensorTable
from weldfusion.timestral_sensor_pilot import prepare, score_forecast
from weldfusion.timestral_sensor_pilot import main


def table(n=320, irregular=False):
    timestamps = np.arange(n) * 0.1
    if irregular:
        timestamps[200:] += 1
    return SensorTable(columns=["current", "voltage"], values=np.tile([100., 20.], (n, 1)),
                       dropped_columns=[], timestamps_s=timestamps)


def test_future_rows_are_hidden_from_forecaster():
    history, observed, meta = prepare(table(), ["current", "voltage"], horizon=32, context=256)
    assert history.shape == (2, 256) and observed.shape == (2, 32)
    assert meta["forecast_start_elapsed_s"] == pytest.approx(28.8)
    with pytest.raises(ValueError, match="irregular"):
        prepare(table(irregular=True), ["current"])
    with pytest.raises(ValueError, match="unavailable"):
        prepare(table(), ["force"])


def test_forecast_metrics_compare_only_held_out_rows():
    history, observed, _ = prepare(table(), ["current"], horizon=32, context=256)

    class FakeForecaster:
        def forecast(self, context, prediction_length):
            assert context.shape == (1, 1, 256) and prediction_length == 32
            assert np.array_equal(context[0], history)
            median = np.full((1, 1, 32), 100.)
            return SimpleNamespace(median=median,
                                   quantile=lambda q: median + (-2 if q == .05 else 2))

    result = score_forecast(FakeForecaster(), history, observed, ["current"])
    assert result["current"]["mae"] == 0
    assert result["current"]["fraction_outside_90pct_interval"] == 0


def test_cli_dry_run_validates_local_csv_without_weights(tmp_path):
    csv = tmp_path / "synthetic.csv"
    csv.write_text("Date,Time,current,voltage\n" + "".join(
        f"04/01/2023,12:00:{i//10:02d}.{i%10}00000,100,20\n" for i in range(300)))
    output = tmp_path / "report.json"
    main([str(csv), str(output), "--columns", "current", "voltage", "--dry-run"])
    import json
    report = json.loads(output.read_text())
    assert report["status"] == "validated_only" and report["source_rows"] == 300
    assert report["cross_stream_alignment_verified"] is False
