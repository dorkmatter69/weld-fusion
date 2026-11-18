# Bounded SlowFast video benchmark

This runs the paper's MMAction2 v0.15.0 SlowFast R101/R50 checkpoint on the
private pilot AVI. It reads 16 overlapping 64-frame windows by default, samples
32 frames per window as in the pretrained model's validation recipe, and
checks that each output has 2,304 features. The source video's measured 1,119
frames imply 1,056 full stride-one windows; the script extrapolates processing
time from the small sample. It is a **throughput estimate**, not a feature
export, anomaly result or full reproduction. It uses the source model's
single center crop; the paper does not fully specify preprocessing, so record
this choice when comparing to its results.

The job is limited to one A10 container, 32 windows at most, a 600-second
function timeout, and no retries. Container startup/image building may take
longer than the function timeout; Modal can bill for time outside the measured
model execution. The source checkpoint and dependencies are downloaded to the
container image, not to the private bucket.

## Bucket access

The Modal account token authenticates *to Modal only*. The worker also needs
read access to the private pilot path. Set up a GCS HMAC key for a dedicated service
account limited to reading the required bucket objects, then create a Modal
Secret named `weldops-pilot-gcs` with exactly these keys:

- `GOOGLE_ACCESS_KEY_ID`
- `GOOGLE_ACCESS_KEY_SECRET`

Use the GCP and Modal web consoles to enter the credentials; never paste them
into chat, the repository, shell history, or a checked-in `.env`. The mount is
read-only and rooted at the `WELDOPS_GCS_PREFIX` value provided locally.
The existing VM service account's bucket-level objectViewer role is broader
than an ideal pilot-only grant; create a dedicated principal or condition before
using it with Modal. Review Intel's research-use terms and GCP-to-Modal data
transfer before enabling access.

## Run

From the repository root, after the script is available in the canonical
checkout and the Modal Secret exists, set `WELDOPS_SAMPLE_ID`,
`WELDOPS_GCS_BUCKET`, and `WELDOPS_GCS_PREFIX` in your local shell (the prefix
must end in `/`). Then run:

```bash
.venv/bin/modal run scripts/modal_slowfast_pilot.py --max-windows 16
```

The function returns counts, checkpoint identity, model load time, decode,
preprocessing and GPU inference times, and an estimated GPU charge for one
weld. It returns no source frames or feature vectors. A failed image build or
missing secret is a setup result, not a measurement; do not extrapolate it.
Only expand beyond the pilot after inspecting the result and costs.

At the published A10 rate of $0.000306 per GPU-second, 10 minutes of GPU time
is about $0.18. CPU, memory, image build, GCS transfer and possible regional
charges are extra. The script's estimate uses that GPU rate only and is not a
spending ceiling. See https://modal.com/pricing for current rates.

## Known limits

- This source environment cannot reach Modal's server; image building and the
  GPU execution have not been validated here.
- The MMAction2 0.15 stack is old. Its pinned CUDA/PyTorch/MMCV wheel set may
  need repair in the remote image. Preserve the checkpoint and record any
  version change before treating the result as paper-comparable.
- One video gives no estimate of how different resolution/duration sessions
  affect throughput. The benchmark uses a single crop and starts at frame 0;
  it does not measure validation AUC, training or three-stream alignment.
