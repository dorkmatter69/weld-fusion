# Timestral / TimesFM sensor-only pilot

This is a bounded **forecasting smoke test** on one authorized sensor CSV. It
predicts the last 32 rows of the *steady arc-on* segment from the preceding 176
steps using two pretrained forecasters, Mistral's Timestral-5M and Google's
TimesFM 3.0, and compares them with two naive baselines (repeat the last value;
repeat the history mean). It reports MAE and 10th–90th percentile interval
coverage against the real held-out readings. It does not produce defect
detection AUC, fit a threshold, access audio/video, or certify clock
synchronization.

TimesFM 3.0 weights are released under a non-commercial, non-production licence
(TimesFM 2.5 weights are Apache-2.0); review both models' terms before use.

## Run on Modal (no local data)

With the `WELDOPS_*` variables used by the other pilots:

```bash
.venv/bin/modal run scripts/modal_timestral_pilot.py --dry-run   # preflight only
.venv/bin/modal run scripts/modal_timestral_pilot.py              # both models, all cadences
```

The launcher signs a one-hour GET URL for `pilot/<sample>.csv`, and a CPU
Sandbox (weights baked into the image) prints aggregate metrics only. The report
is written to `~/weldops-private/timestral/`; nothing is uploaded.

### Cadence modes

The models receive values without timestamps and so assume equal steps. The
pilot CSV's gaps alternate between about 0.100 s and 0.116 s (none missing), so:

- `strict` uses rows as-is only if every gap is within 25% of the median
  (this pilot fails it);
- `rows` treats each row as one step;
- `resample` interpolates the history onto an even grid, refuses gaps over two
  grid steps (possible missing rows), and interpolates the forecast back to the
  real held-out timestamps. Scoring never uses interpolated "truth".

### Pilot result (04-01-23-0024-00, 2026-11-20)

Arc on 6.1–32.2 s; window rows 64–282 (1 s trimmed each end); held out
27.8–31.2 s. Earlier runs that held out the file's last 32 rows mostly forecast
the arc switching **off**. MAE in original units:

| Model | Cadence | Current MAE (A) | Voltage MAE (V) | Outside 80% interval (I / V) |
|---|---|---|---|---|
| Repeat last value | — | 3.31 | 0.357 | — |
| Repeat history mean | — | 6.59 | 0.439 | — |
| Timestral-5M | rows | 3.24 | 0.227 | 19% / 6% |
| TimesFM 3.0 | rows | 3.64 | 0.245 | 22% / 19% |
| Timestral-5M | resample | 3.43 | 0.238 | 28% / 34% |
| TimesFM 3.0 | resample | 3.78 | 0.224 | 34% / 28% |

Neither model beats "repeat the last value" on current; both are about a third
better on voltage. Resampling did not help on this weld. Thirty-two
autocorrelated rows of one weld cannot rank the models or modes.

## Obtain the inputs

The model [package](https://github.com/mistralai/timestral) is public. The
[5M weights](https://huggingface.co/mistralai/Timestral-5M) download once from
Hugging Face and are released separately under a Modified MIT license. Review
the weight terms before use. A CPU is sufficient for this small sample.

Use an authorized machine with access to your private GCP project. Configure
`gcloud auth login` locally if its token has expired. List the project's buckets,
navigate to the approved pilot weld folder, then copy **only its CSV** to a
private directory outside the Git checkout:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud storage buckets list --project YOUR_PROJECT_ID
gcloud storage ls gs://YOUR_BUCKET/path/to/approved/pilot/
mkdir -p "$HOME/weldops-private"
gcloud storage cp gs://YOUR_BUCKET/path/to/approved/pilot/one-weld.csv \
  "$HOME/weldops-private/"
```

Substitute the actual bucket and CSV object returned by `gcloud storage ls`;
the example object path is a placeholder. Do not copy the 40 GB archive for
this one-weld smoke test. If `gcloud storage ls` reports access denied, confirm
the signed-in identity and ask the project's administrator for read access to
the approved pilot object; do not create service-account keys.
Keep the dataset's research-use terms and any local data handling requirements.
Never put raw CSV, derived forecasts, tokens or signed URLs into the public PR.

For an Apple silicon Mac, use Python 3.11 or 3.12 for the standalone pilot:

```bash
cd ~/Developer/weld-twin
git fetch origin feat/overlap-temporal-embedding-review
git switch feat/overlap-temporal-embedding-review  # if already present locally
git pull --ff-only
python3.12 -m venv .venv-timestral
.venv-timestral/bin/python -m pip install -U pip
.venv-timestral/bin/python -m pip install numpy torch
.venv-timestral/bin/python -m pip install 'git+https://github.com/mistralai/timestral.git'
```

The loader in this repository uses NumPy and local CSV files. If Python 3.12
is unavailable, install it first or use a supported Python 3.11 interpreter.
The official Timestral install requires Python >=3.10 and PyTorch >=2.4.
If the branch is not present locally, use
`git switch --track origin/feat/overlap-temporal-embedding-review` instead.

## Local run (alternative)

Replace the placeholder with the authorized local CSV path. The channel names
below were observed in the project's CSV loader tests; the preflight prints
available headers if they differ in your file.

```bash
CSV=/private/path/to/one-weld.csv
PYTHONPATH=src .venv-timestral/bin/python -m weldfusion.timestral_sensor_pilot \
  "$CSV" /private/path/to/preflight.json \
  --columns 'Primary Weld Current' 'Secondary Weld Voltage' --dry-run
PYTHONPATH=src .venv-timestral/bin/python -m weldfusion.timestral_sensor_pilot \
  "$CSV" /private/path/to/timestral-sensor-report.json \
  --columns 'Primary Weld Current' 'Secondary Weld Voltage'
```

Preflight checks the CSV's recorded Date/Time, positive and approximately
regular cadence, finite selected channels, and enough observations. A cadence
gap or irregular stream stops the run rather than silently inventing samples.
The CLI defaults to `--window arc_on --context 176 --cadence strict rows resample`
and `--models timestral`; add `--models timestral timesfm` after installing
`timesfm==3.0.2`. Each channel is forecast independently; the held-out rows are
used **only after** inference to compute MAE and interval coverage. Results are per channel in the CSV's original units. A low error in
one weld does not establish anomaly detection or generalization.

Next, audit the full manifest and session splits, establish a validation
threshold on Good/defect welds, and evaluate untouched test welds against the
paper-style anomaly baselines. Audio/video conditioned forecasting requires
independently reviewed cross-stream clocks and a separate leakage check.
