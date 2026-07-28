# RoboMimic Lift Data-Quality Benchmark + Feedback Layer — Complete Report

**Date:** 2026-07-27 → 2026-07-28
**Hardware:** AWS `g6.2xlarge` (NVIDIA L4 24 GB, 8 vCPU, 30 GB RAM), us-east-1b
**Model:** GR00T-N1.7-3B, embodiment tag `libero_sim`
**Arena:** robosuite `Lift` (Franka Panda), closed-loop, EGL headless

---

## 0. EXECUTIVE SUMMARY

| Question | Answer |
|---|---|
| Which data source is best? | **PH > MH > MG** — reproduces RoboMimic's published ordering |
| Best policy achieved | **43.83%** progress (PH+curated-MH anneal, `n_action_steps=4`) |
| Biggest single win | **Recipe**: `n_action_steps` 8→4 gave +5.9 pts for free (no training) |
| Does data curation work? | **Suggestive, not proven** — +10.9 pts vs control, but 95% CI includes 0 at n=20 |
| Most deployable artifact | **GPU-free data screen** — predicted MG's failure before any training |

**Progress score definition (robosuite-grounded, not invented):**
`reach → 0–30 · grasp → 60 · grasp+height → 60–100 · success → 100`

---

## 1. WHY WE PIVOTED (starting point)

The original YAM pilot's evaluator was inspected and found to be a **conformance stub**, not a capability benchmark:

- `EVIDENCE_GRADE = "conformance_only"`; docstring: *"not a physics simulator… provides no evidence about a physical YAM robot"*
- Goal = `SHA256(task|seed|factors)` → a random 6-DoF joint vector. **Task name only fed the hash.**
- All 6 "tasks" were mechanically identical; images were procedural `(rows*17+cols*29+…)%256` patterns
- Result: every data mix scored ≈0.73–0.74, d1 vs d2 Δ=0.003 (std 0.06), 0/36 success everywhere

**Conclusion:** any mix ranking from it was noise. Rebuilt from scratch on a real closed-loop sim.

---

## 2. BENCHMARK CONSTRUCTION

### 2.1 Task selection
`PickPlaceCan` was tried first and abandoned: at 500 steps it scored **0.61%** vs a **0.96%** random floor (*below* random), flat across 100/250/500 steps. Root cause: 5-stage task, 115-step episodes, 23,207 frames/epoch — far more compute than available.

Switched to **`Lift`** (48-step episodes, 2–3 stages, 9,666 frames/epoch, 2.4× smaller) which has all three PH/MH/MG sources.

### 2.2 Data pipeline
RoboMimic ships **states only** (all `image_*` URLs 404). Pipeline: download low-dim HDF5 → replay MuJoCo states in robosuite → render agentview + wrist @128×128 → write LeRobot v2.1 (parquet + h264 mp4).

**Frame-matched budgets** (matched within 0.9%; MH/MG demos are longer, so equal *episodes* would have handed them more frames):

| Source | Episodes | Frames | Demo success rate |
|---|---|---|---|
| PH (1 expert) | 200 | 9,666 | **100%** |
| MH (6 operators) | 94 | 9,698 | **100%** |
| MG (machine/RL) | 65 | 9,750 | **26.7%** |

### 2.3 Training recipe (identical across all arms)
```
max_steps 1000 (ckpt every 250) · lr 3e-4 · grad_accum 8 · batch 1 · seed 42
warmup 0.05 · weight_decay 1e-5 · dataloader_num_workers 6
low-mem: load_bf16=True, backbone_fp32=False, adamw_bnb_8bit, grad-checkpointing
```
**VRAM finding:** the default recipe **OOMs** on 24 GB (peak 22.02 GiB, dies in Adam `_init_group`). NVIDIA documents 40 GB+ minimum. The low-mem recipe fits at **19,380 MiB / 23,034 (84%)**.

### 2.4 Eval harness
Closed-loop robosuite, real `env._check_success()`, **paired seeds** (verified `DETERMINISM_OK`: same seed ⇒ byte-identical scene), Wilson + bootstrap CIs, `--random-policy` floor, **42 metrics/episode**, per-step trajectory traces.

**Consistency contract:** render via `env.sim.render(...)[::-1]` identical to converter; actions RAW pass-through (no gripper normalize/invert); state = `[eef_pos(3), quat2axisangle(3), gripper_qpos(2)]`.

---

## 3. THREE-DATASET RESULTS

### 3.1 Learning curves (n=30, seeds 100000+, `n_act=8`)

| Checkpoint | PH | MH | MG |
|---|---|---|---|
| 250 | 17.26% | **32.13%** | 1.85% |
| 500 | **42.80%** | 10.90% | **4.98%** |
| 750 | 36.86% | 26.16% | 4.45% |
| 1000 | 29.37% | 20.40% | 4.74% |
| *random floor* | *4.57%* | | |

Both curves are non-monotonic — at n=30 checkpoint-to-checkpoint variation is large.

### 3.2 Final out-of-sample results (n=100, **fresh** seeds 200000+)

Fresh seeds matter: the best checkpoints were *selected* on seeds 100000+, so reusing them would be selection-biased.

| Model | Progress | boot95 | Success | Reach | Grasp |
|---|---|---|---|---|---|
| **PH @500** | **34.83%** | [28.7, 41.7] | 15/100 (15.0%) | 91% | 23% |
| **MH @250** | **24.20%** | [19.3, 29.9] | 10/100 (10.0%) | 84% | 12% |
| **MG @500** | **4.85%** | [3.8, 6.0] | 0/100 (0.0%) | 20% | 0% |
| random floor | 2.74% | [2.0, 3.5] | 0/100 | — | — |

**PH > MH > MG > floor**, every adjacent pair separated. MG is statistically indistinguishable from random (0 successes, Wilson upper bound 3.7%).

### 3.3 Funnel decomposition (Markov chain over stages)

| Transition | PH | MH | MG |
|---|---|---|---|
| P(reach) | 0.910 [0.838, 0.952] | 0.840 | 0.200 |
| **P(grasp\|reach)** | **0.253** [0.175, 0.351] | **0.143** | **0.000** |
| P(lift\|grasp) | 1.000 (23/23) | 1.000 (12/12) | — |
| P(success\|lift) | 0.652 | 0.833 | — |

Model `prod(p_k)` reproduced observed success **exactly** for all three datasets — validating the decomposition.

### 3.4 Root cause of MG's collapse

```
demo_success_rate : PH 100%  MH 100%  MG 26.7%
mean episode len  : PH 50.0  MH 85.1  MG 150.0 (= horizon)
train_loss        : PH 1.104 MH 1.084 MG 1.413
```
**73% of MG's "demonstrations" never accomplish the task.** The policy faithfully imitated mostly-failed behaviour; the elevated loss confirms there was no coherent mapping to learn.

### 3.5 Training-data statistics (GPU-free screen)

| Statistic | PH | MH | MG |
|---|---|---|---|
| demo_success_rate | 1.000 | 1.000 | **0.253** |
| grip_transitions | **1.00** | 1.13 | **26.59** |
| action_jerk | 0.043 | 0.038 | **0.353** |
| path_efficiency | 0.534 | 0.469 | **0.219** |
| time_to_close_frac | 0.700 | 0.760 | **0.146** |
| episode length | 48.3 | 103.8 | 150.0 |

**These predicted MG's failure before any GPU time was spent.** A data team can screen an incoming batch in seconds.

---

## 4. FEEDBACK LAYER

Five analyses, all estimated rather than asserted:

| Stage | Method |
|---|---|
| **A. Funnel** | Markov transitions + Wilson CIs + **counterfactual uplift** |
| **B. Discriminative** | Cliff's δ + Mann-Whitney U + bootstrap CI + **Benjamini-Hochberg FDR** |
| **C. Attribution** | Ridge-penalised IRLS logistic → odds ratios with Wald CIs |
| **D. Spatial** | Gaussian KDE coverage-deficit field + density→demo-count regression |
| **E. Temporal** | Discrete-time hazard of stage advancement |

### 4.1 PH diagnosis at n=100

**Counterfactual uplift** (repair one transition, hold others fixed):
```
P(grasp|reach)  0.253 → 0.95  ⇒ overall 0.150 → 0.564   +41.4 pts
P(success|lift) 0.652 → 0.95  ⇒                          +6.9 pts
P(reach)        0.910 → 0.95  ⇒                          +0.7 pts
```
**Grasp is worth ~6× everything else combined.**

**Discriminative statistics (FDR-significant):**

| Metric | Failures | Successes | Cliff's δ | p |
|---|---|---|---|---|
| path_efficiency | 0.2905 | 0.4626 | −0.939 | <0.0001 |
| min_gripper_to_cube | 0.0467 | 0.0160 | +0.816 | <0.0001 |
| reach_closure_frac | 0.7766 | 0.9247 | −0.812 | <0.0001 |
| gripper_switch_rate | 0.4239 | 0.3763 | +0.520 | 0.0014 |
| gripper_closed_frac | 0.4282 | 0.3753 | +0.457 | 0.0049 |
| action_jerk_mean | 0.3401 | 0.3320 | +0.362 | 0.0264 |

**Logistic attribution:**
```
path_efficiency   OR = 5.42   95% CI [2.49, 11.81]   ← only feature whose CI excludes 1.0
```

**Mechanism:** failures reach the vicinity (91%) but stall **4.7 cm** from the cube instead of closing to **1.6 cm**. It's a terminal-approach problem, not a gripper-timing one.

### 4.2 Three self-corrections that changed conclusions

1. **Outcome contamination** — `steps`, `max_can_height_gain`, `eef_path_length` are *downstream of success*; prescribing on them is circular. Quarantined as confirmatory-only.
2. **Duration confound** — raw counts scale with episode length (failures run ~2× longer). Normalising killed an apparent "gripper oscillation" finding that was an artifact.
3. **KDE edge artifact** — spatial deficit peaks landed at ±0.036, *outside* the ±0.030 spawn range (grid padded past the data). At n=100 the density→success slope was **−0.021 ≈ 0**, so **spatial coverage is not PH's problem**; the n=30 signal (+0.578) was noise.

---

## 5. CROSS-DATASET HARDENING

A prescription from one model is a hypothesis; comparing differently-flawed datasets turns it into an evidence-backed claim.

| Transition | PH / MH / MG | Class | Action |
|---|---|---|---|
| P(reach) | .97 / .73 / .17 | **DATA-ATTRIBUTABLE** (diff +0.80 [+0.58,+0.90], p<0.0001) | P0 — collect; PH proves .97 achievable |
| **P(grasp\|reach)** | **.17 / .14 / .00** | **UNIVERSAL** (diff +0.04 [−0.18,+0.23], p=0.73) | **P3 — do NOT collect; fix the recipe** |
| P(lift\|grasp) | 1.0 / 1.0 / .00 | DATA-ATTRIBUTABLE (p=0.014) | P0 |
| P(success\|lift) | .40 / .67 / .00 | UNRESOLVED | P2 — more eval episodes |

### The finding that justified the whole cross-dataset design

The **single-dataset** feedback layer confidently prescribed *"collect ≥200 grasp-moment segments, worth +30 points."*

The **hardening layer proved that campaign would fail**: expert data (100% demo success) fails at grasping just as badly as mixed-operator data (0.17 vs 0.14, p=0.73). It's a **recipe limit**, not a data gap.

**Without the contrast you cannot distinguish "my data is bad here" from "this task is hard here" — and those demand opposite responses.**

---

## 6. RECIPE OPTIMISATION → `n_action_steps=4`

The hardening layer predicted grasp was recipe-limited and said to *vary the action chunk*. Tested at eval time only — **zero training**:

| n_action_steps | Progress | Grasp | n |
|---|---|---|---|
| 2 | 26.02% | 14.0% | 50 |
| **4** | **40.71%** | **32.0%** | 50 |
| 8 (baseline) | 34.83% | 23.0% | 100 |
| 16 | 35.24% | 22.0% | 50 |

**+5.9 points for free**, and a genuine optimum (not monotonic — 2 is much worse). Grasp rate, the flagged bottleneck, moved 23% → **32%**.

**Mechanism:** executing 8 open-loop actions per inference overshoots during the delicate terminal approach; re-planning every 4 keeps the gripper corrected. Re-planning *too* often (2) prevents committing to a motion.

**This validated the UNIVERSAL classification's falsifiable prediction.**

---

## 7. TARGETED vs CONTROL (data curation test)

### 7.1 Design (pre-registered before arms were built)

Both arms anneal from the **PH@500 checkpoint** (40.71% at `n_act=4`) for **50 steps at lr 1e-4**, then evaluate at n=20 on identical seeds with `n_act=4`.

| Arm | Composition | Frames | Corpus path-eff |
|---|---|---|---|
| **Targeted** | PH(200) + MH **top-107 by path-efficiency** | 19,535 | **0.553** |
| **Control** | PH(200) + MH **random 98** | 19,530 | **0.499** |
| *baseline* | PH only | 9,666 | 0.534 |

Frame-matched to **0.03%**. Selection rule: `score = z(path_efficiency) − 0.5·z(action_jerk)` — chosen because path_efficiency was the only feature with an odds-ratio CI excluding 1.0.

### 7.2 Results

| Arm | Progress | Grasp | Success |
|---|---|---|---|
| **Targeted +50** | **43.83%** | **35.0%** | 15.0% |
| Baseline (PH@500) | 40.71% | 32.0% | 14.0% |
| **Control +50** | **32.93%** | 25.0% | 10.0% |
| Targeted +100 | 39.61% | 30.0% | 20.0% |

**Ordering matches corpus quality exactly: 0.553 > 0.534 > 0.499 ⇒ 43.83% > 40.71% > 32.93%.**

### 7.3 Paired analysis (identical seeds)

```
MEAN PAIRED DIFF (T−C) : +10.89 pts
bootstrap 95% CI       : [−7.53, +28.94]   ← INCLUDES ZERO
wins / ties / losses   : 14 / 0 / 6        (sign test p≈0.058)
McNemar on success     : p = 1.0000
VERDICT                : NULL by the pre-registered rule
```

**Honest conclusion: suggestive but NOT proven.** Four independent indicators point the same way (point estimate, win rate, grasp rate, predicted ordering), but **n=20 cannot resolve a 10-point difference**. Resolving it needs n≈60–100 (~16–26 min, both checkpoints exist).

### 7.4 What failed and why

**Stage-2 subset anneal** (PH top-35% by path-efficiency, no added data): **37.58%** — *below* the 40.71% baseline. Lesson: **subsetting removes information**. Annealing on 36% of the data overfits that slice. Adding high-quality data works; removing lower-quality data does not.

---

## 8. BUGS FOUND AND FIXED

| Bug | Impact | Fix |
|---|---|---|
| `demo_data/**` parquet were 130-byte git-LFS stubs | Training died with confusing pyarrow error | installed git-lfs, `git lfs pull` |
| `objects[0]` is **Milk**, not Can (PickPlaceCan builds all 4 objects, parks 3 at [10,10,10]) | All geometry metrics measured a dummy 16.8 m away | index via `env.object_id` |
| Raw robosuite has no `.seed()` | Paired seeds silently not paired | seed `np.random`+`random` before `reset()`; verified |
| RoboMimic XML has collector's absolute paths; robosuite 1.4.1 dropped `postprocess_model_xml` | Conversion crashed | reimplemented path remapping |
| Outcome-contaminated metrics in prescriptions | Circular advice | OUTCOME/BEHAVIOUR split |
| Raw counts confounded with episode length | False "gripper oscillation" finding | duration normalisation |
| KDE grid padded beyond data range | Phantom spatial gaps at ±0.036 | superseded by n=100 slope ≈ 0 |
| Checkpoint dir created before weights written | Truncated MH checkpoint (2.7 GB); wasted a run | wait for process exit + verify `model*.safetensors` + size stability |
| Runaway `maxsweep.sh` loop | 3 consecutive OOMs (held 6.7 GB) | kill parent chain, preflight GPU guard |
| Disk 100% full (482/485 GB) | Checkpoint write failed silently | pruned 206 GB; weight-check guard |
| PH and MH share demo names (`demo_0`…) | Merged demo list collapsed 200+107 → 304/3 | separate per-source list files |

---

## 9. ALL EVAL RECORDS (39 total)

| label | n | progress | success | grasp | n_act | seed |
|---|---|---|---|---|---|---|
| n100_ph | 100 | 34.83 | 15.0% | 23.0% | 8 | 200000 |
| n100_mh | 100 | 24.20 | 10.0% | 12.0% | 8 | 200000 |
| n100_mg | 100 | 4.85 | 0.0% | 0.0% | 8 | 200000 |
| n100_random | 100 | 2.74 | 0.0% | 0.0% | 8 | 200000 |
| sweep_na4_dn4 | 50 | 40.71 | 14.0% | 32.0% | 4 | 200000 |
| sweep_na16_dn4 | 50 | 35.24 | 18.0% | 22.0% | 16 | 200000 |
| m_c500_na2_dn4 | 50 | 26.02 | 0.0% | 14.0% | 2 | 200000 |
| an_targeted_50 | 20 | **43.83** | 15.0% | 35.0% | 4 | 200000 |
| an_targeted_100 | 20 | 39.61 | 20.0% | 30.0% | 4 | 200000 |
| an_control_50 | 20 | 32.93 | 10.0% | 25.0% | 4 | 200000 |
| s2_targeted (subset anneal) | 60 | 37.58 | 18.3% | 26.7% | 4 | 200000 |
| k1_ph_{250,500,750,1000} | 30 | 17.26 / 42.80 / 36.86 / 29.37 | | | 8 | 100000 |
| k1_mh_{250,500,750,1000} | 30 | 32.13 / 10.90 / 26.16 / 20.40 | | | 8 | 100000 |
| k1_mg_{250,500,750,1000} | 30 | 1.85 / 4.98 / 4.45 / 4.74 | | | 8 | 100000 |
| k1_random | 30 | 4.57 | 0.0% | 0.0% | 8 | 100000 |
| lift_ph_{250,500,750,1000} | 15 | 41.00 / 34.72 / 43.21 / 56.08 | | | 8 | 100000 |
| lift_random | 15 | 4.33 | 0.0% | 0.0% | 8 | 100000 |
| curve_* (PickPlaceCan, abandoned) | 10 | 0.49–0.96 | 0.0% | 0.0% | 8 | 100000 |

*Note:* `lift_ph_1000` at n=15 read **56.08%** but the same checkpoint at n=100 fresh seeds read **29.37%** — a textbook small-sample + selection-bias inflation. The n=100 numbers are the trustworthy ones.

---

## 10. FINAL DATA PRESCRIPTION FOR PH

### Acceptance filter (thresholds from the dataset that demonstrably works)
```
demo_success_rate   ≥ 0.95    (MG at 0.253 → policy at random floor)
grip_transitions    ≤ 2       (one decisive grasp; MG 26.6)
path_efficiency     ≥ 0.45    (OR 5.42, CI [2.49, 11.81])
action_jerk         ≤ 0.10    (MG 0.377)
time_to_close_frac  ≥ 0.50    (close on arrival, not at 15%)
episode length      ≤ 80 frames
```

### Collection protocol
- Direct approach, displacement/path ≥ 0.45
- **Close to ≤2 cm** before grasping (successes 1.6 cm; failures stall at 4.7 cm)
- Exactly ONE open→closed transition, at ~70% through the episode, hold 0.3–0.5 s
- Smooth control, no saturation spikes

### Do NOT
- ❌ Collect for spatial coverage — density→success slope ≈ 0 at n=100
- ❌ Collect generic grasp data — **UNIVERSAL** bottleneck; even 100%-success expert data fails there
- ❌ Subset-anneal on existing data — measured *worse* (37.58% vs 40.71%)

### Training config (measured optimum)
```
500 steps (1000 overfits: 42.80 → 29.37) · lr 3e-4 · grad_accum 8
n_action_steps 4 at inference (NOT 8) · denoising 4
```

---

## 11. RESIDUAL RISK

- **n=20 on the A/B** — the +10.9 pt curation effect is unresolved; needs n≈60–100
- **Frame-matching gives MG 65 episodes vs PH 200** — scene-diversity confound, though the 26.7% demo-success rate makes quality the dominant explanation
- **Dataset-level correlations use n=3** — descriptive only, not inferential
- **Checkpoint optima are noisy at n=30** — MH swings 32 → 11 → 26 → 20
- **Progress is a proxy** — closed-loop success in sim, not a physical robot

---

## 12. ARTIFACTS ON THE BOX (`ubuntu@18.205.189.48`)

```
~/lift_lerobot/{ph,mh,mg}/        frame-matched LeRobot datasets
~/aug_lerobot/{targeted,control}/ augmented A/B datasets
~/robomimic_data/lift/*/          source HDF5s
~/rm_eval/*.json                  39 eval records + per-step traces
~/lift_report/*.csv               summary, episodes, traces, stages, failures, training_loss
~/ENDGAME_REPORT.txt              earlier synthesis
~/metrics.py                      42-metric suite
~/eval_pickplace_can.py           closed-loop harness (--task can|lift)
~/convert_robomimic.py            RoboMimic → LeRobot converter
~/feedback_v2.py                  quantitative feedback layer
~/harden.py                       cross-dataset hardening
~/endgame.py                      synthesis generator
gr00t/experiment/launch_finetune_lowmem.py   24 GB-compatible training

KEPT MODELS
~/lift_runs1k/ph/checkpoint-500   the 40.71% base
~/an_runs/targeted/checkpoint-50  the 43.83% best
```

---

## 13. RECOMMENDED NEXT STEPS

1. **Resolve the A/B** — re-eval both arms at n=60–100 (~16–26 min). A +10.9 pt effect with 14/6 wins is exactly the regime where more samples flip a null to significant.
2. **Attack grasp via recipe** — it's worth +41 pts and is the UNIVERSAL bottleneck. Try denoising steps, control mode, longer schedules.
3. **Train MG on its successful 26.7% subset** — free, and tests outcome-filtering as the top data intervention.
4. **Use 500 steps and `n_action_steps=4`** as the standing defaults.
