# Timestral sensor-only pilot

This is a bounded **forecasting smoke test** on one authorized sensor CSV. It
predicts the last 32 recorded rows from the preceding 256 rows using Mistral's
pretrained Timestral-5M. It compares the forecast median and 5th–95th percentile
interval against held-out rows. It does not produce defect detection AUC, fit a
threshold, access audio/video, or certify clock synchronization.

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

## Preflight and sample forecast

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
The model receives `(1, 2, 256)` historical rows and forecasts 32 future rows;
the held-out rows are read **only after** inference to compute MAE and interval
coverage. Results are per channel in the CSV's original units. A low error in
one weld does not establish anomaly detection or generalization.

Next, audit the full manifest and session splits, establish a validation
threshold on Good/defect welds, and evaluate untouched test welds against the
paper-style anomaly baselines. Audio/video conditioned forecasting requires
independently reviewed cross-stream clocks and a separate leakage check.
