# Intel dataset: full staging and the paper's audio/video baselines

Stemmer et al. 2024, [Unsupervised Welding Defect Detection Using Audio and
Video](https://arxiv.org/abs/2409.02290). Paper test AUCs: audio 0.8460 (Table 9),
video 0.8977 (Table 11, max over 2 s moving average), late fusion 0.9178 (Table 12).

The Intel Robotic Welding Multimodal Dataset is licensed for **research use only**.
This page reports aggregate results only. Media, per-weld scores, feature files
and signed URLs stay in private GCP storage or `~/weldops-private/`, never in Git.

## Staging (done 2026-11-20)

`scripts/build_gce_stage_startup.py` builds a startup script for a temporary
Compute Engine VM in the bucket's region. The VM copies the archive, extracts it
with `weldfusion.archive`, which rejects unsafe paths and hashes every member,
and uploads the tree to `staged/intel-welding/` in the private data bucket. The
VM is then deleted. Media never left GCP.

- 32,307 files, 44.7 GB uncompressed. Every manifest weld has its AVI, FLAC and
  CSV; 3 welds have no post-weld images.
- Modal reads through the existing signer service account. It holds
  `objectViewer` on the data bucket, restricted by an IAM condition to
  `staged/intel-welding/` (reads of `raw/` return 403). The launcher signs
  GET URLs locally and sends them on the Sandbox's stdin. No keys are created.

## Manifest and split audit

4,040 welds, 236 sessions; each session holds one category.

| | TRAIN | VALIDATION | TEST |
|---|---|---|---|
| Good | 576 | 122 | 121 |
| 11 defect categories (80–240 each) | 0 | 1,610 | 1,611 |

**The published split is by weld, not by session.** 216 of 236 sessions have
welds in more than one split. 72 of the 121 Good test welds come from sessions
that also supply Good training welds. A Good-only autoencoder has therefore seen
the exact setup of most Good test welds. Because each session has a single
category, a model can also separate categories by recognising session conditions.
Every result is reported under two protocols:

- `paper`: the manifest split as published (comparable with the paper);
- `session`: session-disjoint. Good sessions are split 70/15/15 and each defect
  category's sessions are halved between VALIDATION and TEST.
  `weldfusion.audio_experiment.session_resplit` builds it, seed 7.

## Audio autoencoder (`src/weldfusion/audio_experiment.py`)

The network follows Table 5: BatchNorm, 5 Conv1d and 5 ConvTranspose1d layers,
kernel 3, bottleneck 48. Training uses Good TRAIN welds only, MSE, Adam with a
one-cycle schedule peaking at 1e-4, for 50 epochs. The weld score is the mean
frame error. Deviations from the paper:

- **The released audio is 16 kHz; the paper used 192 kHz.** Windows are matched
  in duration (1,365 samples ≈ 85.3 ms, 50% hop), not in bandwidth. Nothing
  above 8 kHz is available.
- The paper does not state the window function, log scaling, batch size or
  padding. We use Hann, log1p, batch 256 or 32, and padding 1.
- The model has 23.4M parameters versus the paper's 31.7M, because there are
  fewer frequency bins.

Overall AUC (paper's FFT/bottleneck setting, single seed):

| Run | Validation | Test |
|---|---|---|
| Paper (Table 9) | 0.8451 | 0.8460 |
| `paper` split, standardized target, batch 256 | 0.681 | 0.666 |
| `paper` split, raw log-magnitude target, batch 256 | 0.673 | 0.660 |
| `paper` split, raw target, batch 32 | 0.690 | 0.674 |
| `session` split, standardized target, batch 256 | 0.544 | 0.637 |

Per-defect test AUC, standardized target, batch 256:

| Defect | Paper | `paper` split | `session` split |
|---|---|---|---|
| Crater cracks | 0.945 | 0.906 | 0.954 |
| Undercut | 0.899 | 0.755 | 0.886 |
| Porosity w/ excessive penetration | 0.966 | 0.786 | 0.784 |
| Burnthrough | 0.702 | 0.754 | 0.788 |
| Excessive convexity | 0.884 | 0.760 | 0.778 |
| Porosity | 0.995 | 0.701 | 0.611 |
| Overlap | 0.974 | 0.609 | 0.409 |
| Excessive penetration | 0.781 | 0.582 | 0.542 |
| Warping | 0.765 | 0.560 | 0.514 |
| Lack of fusion | 0.837 | 0.551 | 0.457 |
| Spatter | 0.780 | 0.526 | 0.508 |

**Interpretation:**
- On the released audio, the paper's audio model reaches about 0.67 test AUC,
  not 0.85.
- Changing the reconstruction target or giving 8× more optimizer steps moved
  the result by about 0.01, so neither setting explains the gap.
- The largest shortfalls are porosity and overlap (paper ≥ 0.97), which is
  consistent with relying on content above 8 kHz.
- The `session` split's lower validation AUC fits the session overlap
  inflating `paper`-split scores. With single seeds and weak models, the
  size of that effect is not yet established.
- Max aggregation gives about 0.48, matching the paper's preference for the mean.
- The paper's other window lengths (21, 171 and 341 ms equivalents) have
  not been tried.

## Video features (`scripts/modal_slowfast_batch.py`)

The exporter uses the same SlowFast R101 checkpoint (MMAction2 v0.15.0) and the
same preprocessing as the pilot export. Each window covers 64 frames, of which
32 are sampled; there is one window per frame. Frames are decoded and cropped
once and windows run in batches. On the pilot weld its features are **identical**
to the earlier one-window-at-a-time export (maximum absolute difference 0.0).
On a 4-weld benchmark it needed about 25 s of A10G time per weld versus about
86 s before. Output is create-only float16 archives in the private pilot bucket.
The full export (about 36 GPU-hours) and the Table 6 video autoencoder have not
been run yet.

## Run

```bash
.venv/bin/modal run scripts/modal_audio_baseline.py --per-cell 3 --epochs 2 --protocols paper  # smoke test
.venv/bin/modal run scripts/modal_audio_baseline.py                                            # both protocols
.venv/bin/modal run scripts/modal_audio_baseline.py --protocols paper --variants raw_b256,raw_b32
.venv/bin/modal run scripts/modal_slowfast_batch.py --limit 4                                 # video benchmark
```

The audio launcher reads the manifest copy at `~/weldops-private/intel/manifest.csv`.
Its tests are in `tests/test_audio_experiment.py`; the torch training test needs torch installed.
