# Data Quality for VLA Fine-Tuning

A controlled study of demonstration quality for vision-language-action fine-tuning,
run on RoboMimic `Lift` with GR00T-N1.7-3B.

**Report:** https://shaikzaidhaaris.github.io/vla-data-quality/

## What it asks

Three recordings of the same task, differing only in who produced them:

| Source | Episodes | Frames | Demos that succeed |
|---|---|---|---|
| PH, one skilled teleoperator | 200 | 9,666 | 100% |
| MH, six operators of varying skill | 94 | 9,698 | 100% |
| MG, an RL agent left to run | 65 | 9,750 | 26.7% |

Budgets are matched on frames rather than episodes, so the comparison measures
quality instead of volume.

## Headline results

| Model | Mean progress | 95% CI | Success |
|---|---|---|---|
| PH | 34.83% | 28.7 – 41.7 | 15 / 100 |
| MH | 24.20% | 19.3 – 29.9 | 10 / 100 |
| MG | 4.85% | 3.8 – 6.0 | 0 / 100 |
| random floor | 2.74% | 2.0 – 3.5 | 0 / 100 |

Best policy reached **43.83%** after annealing on curated additions.
Changing one inference-time setting was worth **+5.9 points** with no retraining.

## Contents

- `index.html` — the report
- `METHODOLOGY.md` — full write-up, every number and every bug found
- `data/eval_records.json` — all 39 evaluation records
- `data/*.csv` — per-episode results, per-step traces, stage histograms, training loss

## Method notes

1,340 evaluated rollouts. Paired seeds throughout, verified deterministic.
Bootstrap intervals for means, Wilson for rates, Benjamini-Hochberg across
comparison families. Progress is scored on robosuite's own staged physics:
reach up to 30, grasp 60, lift interpolates to 100.
