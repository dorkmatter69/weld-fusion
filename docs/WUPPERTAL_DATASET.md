# Wuppertal metal arc welding dataset

Hahn, Königsfeld, Tercan, Buchholz, Purrio, Angerhausen, Meyes, Meisen (2025),
*Metal Arc Welding*, Zenodo, [doi:10.5281/zenodo.15497262](https://doi.org/10.5281/zenodo.15497262),
**CC BY 4.0** (attribution required). One file, `welding_data.csv`, 2.67 GB,
MD5 `63489186e5ddae7fed6941444041cb68`. No data is committed to this repository.

## Run (Modal; nothing downloads to the Mac)

```bash
.venv/bin/modal run scripts/modal_wuppertal_profile.py      # download to Volume, verify MD5, profile
.venv/bin/modal run scripts/modal_wuppertal_order_check.py  # time-order tests + pulse plots
.venv/bin/modal run scripts/modal_wuppertal_baseline.py     # classifier, confound probe, episode export
```

The CSV lives in the Modal Volume `weldops-wuppertal-maw`. Outputs go to
`~/weldops-private/wuppertal/`. Parsing code is in `src/weldfusion/wuppertal.py`.

## Structure (measured 2026-11-20)

- 362,034 rows, no missing values. Columns `V_000..V_199`, `I_000..I_199`,
  `label`, `exp_id`, `welding_run_id`. (The Zenodo page's schema says `labels`
  and `exp_ids`; the file uses the singular names.)
- 4 experiments, 53 runs. Labels apply to **each pulse**: 19% good (1), 25% bad
  (0), 56% unlabeled (−1). Every run contains more than one label.
- Each row is one current pulse: about 45 A background, peak about 520 A, back to
  background. The pulse period is undocumented; exports assume 2 ms (200 samples
  at 100 kHz) and record that assumption.
- **Rows are in time order within a contiguous file block of a run.** In file
  order versus a shuffled copy, label changes are 0.4% as frequent, and the lag-1
  autocorrelation of peak current is 0.79 versus 0.00. The end-to-start join is 0.13× a random
  pair's. Each run is stored as 1–4 blocks ("segments", 136 in total), separated by
  arc-start transients. Segments are never joined.

## Good/bad pulse baseline

Score = P(bad). Pulses of one run never appear in both train and test.

| Model | Held-out experiment AUC | Held-out run AUC | Within-segment AUC (median, 58 segments) |
|---|---|---|---|
| Logistic regression, 14 pulse features | 0.774 | 0.788 | 0.51 |
| Gradient boosting, features + waveform | 0.666 | 0.723 | 0.49–0.56 |
| **Segment-average features only** (shortcut probe) | 0.752 | 0.741 | 0.50 |

**Interpretation:** a model that sees only its segment's *average* features nearly
matches the pulse model. Inside the 58 segments that contain both labels, no model
separates good from bad pulses (AUC ≈ 0.5). 53% of labeled pulses sit in segments
with a single label. The labels therefore mostly mark **which segment or condition**
a pulse came from. Pulse shape did not reveal pulse quality within a condition.
Claims of per-pulse defect detection on this dataset need a within-segment
evaluation. Read the dataset paper for how labels were assigned before going further.

## Shared format and Intel bridge

Each segment is exported as a `WeldEpisode` v1 JSON and MCAP (`/weld/meta`,
`/sensors/weld`, `/weld/events`). Pulses are averaged in windows of 58 (0.116 s,
the Intel pilot's median sensor step). Label stretches become `quality_label`
events, and the timeline marks `pulse_period_assumed: true`. The browser demo's
file picker can open the JSON.

Wuppertal segments at that timescale (median of 129 segments) compared with the
Intel pilot's steady arc:

| | Wuppertal | Intel pilot |
|---|---|---|
| Current mean / SD / lag-1 | 166 A / 3.0 A / 0.69 | 179 A / 5.4 A / 0.51 |
| Voltage mean / SD / lag-1 | 22.5 V / 0.42 V / 0.51 | 17.3 V / 0.36 V / 0.67 |

At this timescale the current behaves similarly. Voltage is about 5 V lower in the
Intel data, probably because it is measured at a different point. Compare relative
changes, not raw volts. The Intel readings are tight and unimodal (179 ± 5 A),
so they behave like averages of a pulsed waveform rather than instantaneous samples.
