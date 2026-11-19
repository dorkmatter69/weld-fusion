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

The job is limited to one A10 Sandbox, 32 windows at most and a 600-second
Sandbox lifetime. Container startup/image building may take longer than the
Sandbox timeout; Modal can bill for time outside the measured model execution.
The source checkpoint and dependencies are downloaded to the container image,
not to the private bucket. Modal's 2026.10 builder cannot add Python 3.8 to
a Function, so the isolated GPU Sandbox uses the official legacy PyTorch image
with its bundled Python.

## Bucket access without service-account keys

GCP's `iam.disableServiceAccountKeyCreation` policy prevents HMAC keys. Do not
weaken it. The dedicated pilot bucket contains only the pilot AVI; its signer
service account has read access to that bucket only. The local entrypoint uses
`gcloud storage sign-url` and IAM's managed signing key to make a one-hour GET
URL for that object. It injects that URL into a single ephemeral Modal Sandbox
as a runtime Secret, which downloads no more than 100 MiB and deletes the
temporary AVI when done.
Do not paste the signed URL into chat or logs: anyone holding it can read that
one file until expiry. No GCS or service-account key is stored in Modal.

The user account running `gcloud` needs `roles/iam.serviceAccountTokenCreator`
on the dedicated signer (not on the whole project). The pilot bucket and this
IAM binding were created for the one-weld benchmark. Review Intel's research-use
terms and GCP-to-Modal data transfer before execution.

## Run

From the repository root, after pulling the latest branch, set
`WELDOPS_SAMPLE_ID`, `WELDOPS_GCS_BUCKET`, `WELDOPS_GCP_PROJECT`, and
`WELDOPS_SIGNING_SERVICE_ACCOUNT` in your local shell. If your isolated bucket
is not in `us-west1`, also set `WELDOPS_GCS_REGION`. Then run:

```bash
.venv/bin/modal run scripts/modal_slowfast_pilot.py --max-windows 16
```

The local entrypoint signs the URL immediately before invoking Modal and never
prints it. The Sandbox returns counts, checkpoint identity, model load time, decode,
preprocessing and GPU inference times, and an estimated GPU charge for one
weld. It returns no source frames or feature vectors. A failed image build or
missing secret is a setup result, not a measurement; do not extrapolate it.
Only expand beyond the pilot after inspecting the result and costs.

## Confirmed pilot result

The local command and GPU worker completed successfully on the user's Modal
account. The worker processed 16 windows from the 1,119-frame, 31 fps pilot
video and confirmed a 2,304-dimensional feature vector per window. It reported
1.329 seconds of GPU inference, 0.554 seconds of preprocessing, 2.349 seconds
to download the AVI, and 4.982 seconds to load the model. Its stride-one
extrapolation for all 1,056 windows was 133.4 seconds of measured-category
compute and about $0.041 of A10 GPU time, excluding startup, image build,
CPU, memory and GCS transfer. This is an estimate from the first 16 windows
of a single video, not a full-weld measurement. A previous worker run reported
144.6 seconds and $0.044 on the same sample; the variation underscores that
the extrapolation is approximate.

## Export all pilot video features

The isolated pilot bucket's signer also has the create-only
`roles/storage.objectCreator` permission on that bucket. It has no
overwrite or delete permission. The export signs an upload for
`features/slowfast-r101-v015/<pilot-id>.npz` with a generation-match-zero
precondition; an existing object makes the job stop. No service-account key
is involved.

With the same local `WELDOPS_*` variables used for the benchmark, run:

```bash
.venv/bin/modal run scripts/modal_slowfast_export.py
```

This job is restricted to the measured 1,119-frame pilot and processes all
1,056 overlapping windows. It writes a compressed NumPy archive containing
`features` (float32 [1056, 2304]), `window_start_frame`,
`window_end_frame`, `window_center_seconds`, frame rate, frame count,
sample ID, checkpoint, and MMAction2 version. Center time is the midpoint
of each 64-frame window relative to the start of the AVI. The worker validates
the array and uploads it directly to the isolated GCS bucket. The local
entrypoint then checks the uploaded object's size and GCS MD5 metadata
against the worker's values. The worker
also reports a SHA-256 digest for provenance. Only metadata and timings
return to the local terminal. The source AVI never downloads
to the Mac. The export has a 600-second Sandbox lifetime, one A10, and no
automatic retry. Review the result before running additional welds.

### Completed pilot export

The user ran the export successfully and shared the terminal result. The
saved object is `features/slowfast-r101-v015/04-01-23-0024-00.npz`.
It is 8,644,803 bytes and contains 1,056 validated 2,304-dimensional
feature vectors. Its SHA-256, computed by the worker, is
`3591c5c5d58c661a583bcada201fe64135bba03809636fd3416be04aedc5009d`.
The worker's MD5 was `UBsPh/uZivft+85WXbAQpw==`. An independent
GCS metadata read confirmed that MD5, object size, and content type.
The measured categories totaled about 85.9 seconds: 38.707 seconds of
preprocessing, 38.873 seconds of GPU inference, 6.930 seconds of model
loading, 0.717 seconds of video decoding, and 0.697 seconds downloading
the pilot AVI. The script's GPU-only rate calculation was $0.026; this
is not a Modal invoice and excludes other charges. Relative video
window centers span 1.016129–35.048387 seconds. They are not aligned
to sensor or audio clocks.

At the published A10 rate of $0.000306 per GPU-second, 10 minutes of GPU time
is about $0.18. CPU, memory, image build, GCS transfer and possible regional
charges are extra. The script's estimate uses that GPU rate only and is not a
spending ceiling. See https://modal.com/pricing for current rates.

## Known limits

- This source environment cannot reach Modal's server. The user ran the
  benchmark and export successfully on their Modal account and shared
  the outputs.
- The MMAction2 0.15 stack is old. Its pinned CUDA/PyTorch/MMCV wheel set may
  need repair in future environments. Preserve the checkpoint and record any
  version change before treating the result as paper-comparable.
- One video gives no estimate of how different resolution/duration sessions
  affect throughput. The exporter uses a single crop and starts at frame 0;
  it does not measure validation AUC, training or three-stream alignment.
