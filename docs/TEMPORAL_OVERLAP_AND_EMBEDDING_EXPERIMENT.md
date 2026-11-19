# Overlap defect: temporal video ablation and embedding explorer

## Status and evidence

This is an **implemented local research protocol**, not a measured improvement.
Only one weld's SlowFast vectors have been exported to the private GCS bucket;
the complete labeled cohort and locally staged media are not available in this
checkout. No training, all-weld export, or EmbeddingGemma inference has run.
The source dataset remains private; do not commit features, recordings or index
JSON files to Git.

Stemmer et al. [Table 11](https://arxiv.org/pdf/2409.02290) report **Overlap**
test AUC of 0.8516 for the 2-second-smoothed video model; the paper's audio
Overlap AUC is 0.9743 (Table 9), and audio/video late fusion scores 0.9310
(Table 12). The reported values are reference targets, not results from the
released 16 kHz files. Overlap is a *defect class*, distinct from overlapping
video windows. The audio-only number being already high means that a new video
model might improve video AUC without improving overall fusion.

## Hypothesis and preregistered test

**Hypothesis:** modeling the progression of weld appearance beyond each 64-frame
SlowFast window increases held-out **Good-vs-Overlap video AUC** relative to
the paper-style framewise video autoencoder, on the *same* released-dataset
split. The mechanism is temporal context, not a new video backbone.

- Primary outcome: paired difference in test AUC (temporal minus autoencoder)
  comparing all held-out Overlap welds against the same held-out Good welds.
  Report both scores and a paired bootstrap interval; inspect session clustering
  before making a population claim.
- Secondary outcomes: overall Good-vs-any-defect AUC and each of the other ten
  defect-vs-Good AUCs, compute time, memory and inference latency. Report
  validation and test separately. A favorable Overlap result alone is not proof
  of an overall anomaly-detection improvement.
- Fit both models with Good training welds only. Select epochs on validation
  overall AUC; fix architecture, aggregation and seed before opening test.
  Run multiple seeds and report variability before interpreting a small gain.
- Keep an untouched test cohort. Audit `session_id` boundaries, sample IDs,
  exact `CATEGORY` labels, recording conditions, and the feature checkpoint.
  The script rejects a session shared across partitions.
- False-positive behavior should be examined at a threshold selected from
  validation Good welds, with recall on test Overlap and review of false alarms.
  AUC alone says nothing about the chosen operating threshold.

**Null:** long-range context does not improve separation of Good and Overlap,
or a gain disappears across seeds/sessions. The pilot alone cannot test this:
one weld does not provide Good and Overlap test cohorts.

## Two video models

Both receive the same fixed 2,304-value, stride-one SlowFast R101 features.
The baseline uses the paper's Table 6 widths
`2304 → 512 → 256 → 128 → 64 → 64 → 64 → 128 → 256 → 512 → 2304`,
ReLU/dropout and reconstruction MSE. Our implementation is **paper-style**:
paper-specific training schedule, exact regularization placement, and
validation selection must still be audited before claiming a numerical
replication. The script defaults to 20 epochs for smoke runs; the paper says
it trained the video model for up to 1,000 epochs. The transformer uses a
128-wide projection, two four-head encoder blocks and a masked-vector
reconstruction objective (25% masks). It is an ablation, not Intel's model.

Adjacent stride-one features share 63 of 64 raw frames. To prevent a trivial
masked-token shortcut, temporal contexts use 64 interleaved phases: indices
`0,64,128…`, `1,65,129…`, etc. Tokens within a context share no source frames.
All phases are scored once with complementary masks. Training cost grows with
this design, and the temporal model can see distant positions within a weld;
it is **offline** and noncausal. Do not quote a real-time detection latency.
Both models reduce per-window errors to peak of a full 2-second moving mean.

Input JSON example (one row per weld, absolute paths or paths relative to the
index file):

```json
[
  {"sample_id":"session-a/good-001", "session_id":"session-a", "split":"TRAIN", "category":"Good", "feature_path":"features/good-001.npz"},
  {"sample_id":"session-b/good-002", "session_id":"session-b", "split":"VALIDATION", "category":"Good", "feature_path":"features/good-002.npz"},
  {"sample_id":"session-b/overlap-001", "session_id":"session-b", "split":"VALIDATION", "category":"Overlap", "feature_path":"features/overlap-001.npz"},
  {"sample_id":"session-c/good-003", "session_id":"session-c", "split":"TEST", "category":"Good", "feature_path":"features/good-003.npz"},
  {"sample_id":"session-c/overlap-002", "session_id":"session-c", "split":"TEST", "category":"Overlap", "feature_path":"features/overlap-002.npz"}
]
```

These IDs are schema examples, not claimed dataset members. Each `.npz` needs
`features`, `window_center_seconds`, `window_start_frame`, and `sample_id`
from the existing SlowFast exporter. `sample_id` inside `.npz` is the weld
folder stem. A real all-defect cohort is needed for overall AUC and proper
model selection. Run on an authorized worker with adequate storage and GPU:

```bash
python -m pip install -e '.[research,dev]'
PYTHONPATH=src python -m weldfusion.video_experiment cohort.json video_report.json --epochs 20 --device cuda
```

The report includes per-model validation AUC, best epoch, test overall and
per-defect AUC, per-weld scores, and a weld-level paired bootstrap interval for
Overlap. The interval is exploratory if many welds share a session. Review
session-level confidence before a paper claim. For a full baseline, extend the
epoch budget only after a compute benchmark and predeclare the selection rule.
If audio scores are available, join each model's `scores` array by qualified
sample ID and run the existing `weldfusion.anomaly_eval` separately for each;
select each fusion weight on validation and compare test Overlap/overall AUC.
Do not tune a weight on test Overlap.

## EmbeddingGemma 2 branch and dashboard

This branch is for **retrieval and review first**, then a separately evaluated
embedding-based anomaly experiment. Google's
[developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)
describes a shared 768-dimensional video/audio space. Video defaults to 1 FPS;
this pipeline requests 4 FPS over 4-second clips. It still may miss brief
events, so compare retrieval against the original 31 FPS video and SlowFast
scores. Audio is converted to 16 kHz mono. The sensor CSV is not a native
EmbeddingGemma numeric time-series input and stays outside this branch.

Create a local media manifest with the six columns `sample_id`, `session_id`,
`split`, `category`, `video_path`, `audio_path`; paths must point to authorized
local AVI and FLAC files. Run where the private media resides:

```bash
python -m pip install -e '.[research]'
python scripts/build_embedding_index.py media.json embedding_index.json --clip-seconds 4 --hop-seconds 2 --video-fps 4
python -m http.server 8000 --directory web
```

Open `http://localhost:8000/embeddings.html`, load the local index JSON, then
filter by split, category, session or weld ID. Select a segment for nearest
neighbors using the video **or** audio embedding. The browser does not fetch
media or send vectors to a service. The index itself contains private-derived
vectors and labels; keep it in authorized storage. FFmpeg/ffprobe and model
weights must be installed on the worker. The exporter refuses overwrite.
The **Explore synthetic demo** button exercises filtering and similarity UI
with fabricated vectors and labels; it is never a model result.
The index builder defaults to **24 segments maximum** so the first run is a
bounded pilot. Increase `--max-segments` only after measuring time and reviewing
the chosen media cohort; the first rows are processed in manifest order, so
prepare a small, balanced media manifest for any retrieval comparison. This
is not a full-dataset export by default.
For the dataset's research terms, review the authorized compute environment
and transfer route before moving private media.

The script uses equal *file-relative* intervals in the two source files;
these are **not clock-aligned physical moments**. The dashboard explicitly
marks alignment unverified. It does not display synced playback or make
defect predictions. A follow-up retrieval experiment should measure Recall@K
for expert-reviewed Overlap segments, with query and candidate sessions kept
separate, against SlowFast nearest-neighbor retrieval. Only then test an
embedding-based anomaly score on the held-out weld split.
