# Intel audio/video paper: local evaluation stage

This stage accepts one audio and one video anomaly score per weld. It does not
extract features, train models, access GCS, or make a claim of reproducing the
paper's AUC. It can run without NumPy, PyTorch, Modal, or private media.

Paper: Stemmer et al., [Unsupervised Welding Defect Detection Using Audio and
Video](https://arxiv.org/pdf/2409.02290), sections 3–4. The reported test AUCs
are 0.8460 (audio), 0.8977 (video), and 0.9178 (late fusion).

## Score contract

Supply a JSON array with one record per uniquely qualified weld:

```json
[
  {"sample_id":"session-1/good-1","split":"TRAIN","category":"Good","audio_score":0.2,"video_score":0.3},
  {"sample_id":"session-2/good-2","split":"TRAIN","category":"Good","audio_score":0.4,"video_score":0.5},
  {"sample_id":"session-3/good-3","split":"TRAIN","category":"Good","audio_score":0.3,"video_score":0.4},
  {"sample_id":"session-4/good-4","split":"VALIDATION","category":"Good","audio_score":0.3,"video_score":0.4},
  {"sample_id":"session-4/defect-1","split":"VALIDATION","category":"Spatter","audio_score":0.7,"video_score":0.8},
  {"sample_id":"session-5/good-5","split":"TEST","category":"Good","audio_score":0.3,"video_score":0.4},
  {"sample_id":"session-5/defect-2","split":"TEST","category":"Spatter","audio_score":0.8,"video_score":0.9}
]
```

The records above illustrate schema and split boundaries only. They are not
Intel measurements. Run from the repository root:

```bash
PYTHONPATH=src python3 -m weldfusion.anomaly_eval scores.json report.json
PYTHONPATH=src python3 -m unittest discover -s tests -p test_anomaly_eval.py
```

`audio_mean(errors)` averages frame errors. `video_peak_mean(errors,
window_frames)` takes the largest full-window moving mean; use a window of
roughly two seconds based on the actual video timing. The script standardizes
each modality using Good **training** scores, selects the audio fraction on
validation in 0.01 steps, and reports test AUC overall and against the Good
test cohort for each defect. It rejects defects in training and duplicate IDs.

## Replication boundaries

- The paper used 192 kHz audio and the released FLAC is 16 kHz. A time-matched
  FFT window can test the method on the released data, but cannot restore high
  frequencies or give an exact audio replication.
- The paper cites MMAction2 v0.15.0 checkpoint
  `slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth` and
  a fixed 2,304-dimensional feature per 64-frame window. A different
  checkpoint or window stride is a separate experiment.
- The fixed-frame moving mean here needs a documented boundary policy and
  actual frame-timing check before an exact paper comparison. Whole-weld
  fusion does not establish synchronized audio/video frames.
- Audit manifest split and session overlap before training. Do not tune
  hyperparameters or fusion weights using the test cohort.
- The dataset's Intel Research Use License limits this work to research.
  Keep private media in authorized environments; review terms before any
  third-party compute transfer.
