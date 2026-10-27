# Proposal r1: How deep does few-shot TinyTL need to back-propagate?

Flowers102, ProxylessNAS-Mobile + lite residual (ws+gn checkpoint), RTX 3060, PyTorch 2.5.1.

## 0. New feasibility measurement made for this proposal (no training)

The context noted that there was no single memory harness (559.5 vs 573.4 MiB for the same L+B config) and no batch-1 numbers for truncated variants. I wrote one harness and ran it twice in separate processes (about 2 minutes each, memory only; another application was using the GPU at about 40% at the time, so I did not time anything).

- Scripts: `design\mem_harness.py` (all arms, batch 8 and 1, plus a breakdown of what autograd saves) and `design\mem_repeat.py` (second process, arms in reverse order). Raw output: `design\mem_harness_result.json`.
- Method: same build as `probe_depth.py`. Synthetic 224 px fp32 input, Adam, 2 warm-up steps, then 3 full steps (zero_grad, forward, loss, backward, step). Peak is `max_memory_allocated` minus the allocation before `.cuda()`, so it includes 18.2 MiB of fp32 weights. Within a process every arm repeats exactly (spread 0.00 MiB).
- **The 559.5 vs 573.4 discrepancy is explained.** The first arm measured in a process also pays for a one-time workspace of about 16 MiB (cuBLAS/cuDNN), which later arms do not see. L+B measured first gives 573.4; measured later it gives 557.5. FT-Last measured first gives 82.9 (the 82.9 in the context table, because `probe_methods.py` measures `last` first); measured later it gives 67.4. Once the first arm is excluded, the two processes agree within 1.1 MiB. All numbers below are steady-state (not first in process).

**Table M. Measured train-step memory (MiB) and what is trained**

| arm | trained (all arms keep every lite residual in the forward pass) | trainable params | profiler MB (b8) | measured b8 | measured b1 |
|---|---|---|---|---|---|
| FT-Last | classifier only (upstream `last`) | 130,662 | 31.6 | 67.4 | 25.1-25.9 |
| T5 | biases + lite residual from block 21, feature-mix bias, classifier | 362,598 | 32.3 | 68.4 | 30.3 |
| T4 | same, from block 17 (stages 4-5) | 1,113,126 | 34.6 | 105.0-105.2 | 51.5-52.1 |
| T3 | same, from block 13 (stages 3-5, 9 of 22 blocks) | 1,578,246 | 36.0 | 168.6-169.1 | 66.7-67.2 |
| T2 | same, from block 9 | 1,863,766 | 36.8 | 225.7 | 78.5 |
| T1 | same, from block 5 | 1,944,182 | 37.1 | 328.4 | 92.8 |
| T0 | same, from block 1 | 1,962,230 | 37.1 | 470.7 | 112.0 |
| **LB** = TinyTL-L+B | all biases + all lite residuals + classifier | 1,962,310 | 37.1 | **557.5** | **122.1-123.2** |
| TinyTL-B | all biases + classifier | 149,622 | 31.7 | 537.0 | 102.4 |
| FT-Full | everything | 4,746,590 | 390.8 | 598.7 | 160.0 |

Apart from the first-in-process artifact, these agree with the context tables to within 2 MiB (for example L+B 557.5 vs 559.5, TinyTL-B 537.0 vs 538.7).

**What autograd saves for backward** (batch 8, unique storage bytes, grouped by the leaf module that first saved them):

| arm | GroupNorm inputs | activation outputs | first saved by conv (inputs + standardized weights) | total |
|---|---|---|---|---|
| LB | 172.3 | 146.3 | 192.6 | 511.8 |
| TinyTL-B | 172.3 | 146.3 | 185.7 | 504.9 |
| FT-Full | 172.3 | 146.3 | 207.9 | 527.1 |
| T3 | 38.5 | 33.4 | 56.0 | 128.5 |
| T4 | 18.9 | 16.8 | 32.4 | 68.7 |
| FT-Last | 0 | 0 | 0 | 0 |

Reading: even with frozen weights, a bias update has to pass its gradient back through every GroupNorm and activation between the loss and that bias. PyTorch therefore keeps about 505 MiB for TinyTL-B, and the profiler counts none of it (it charges a norm or conv layer only when that layer's own weight is trainable). Part of this is a framework artifact: conv inputs are not needed when the conv weight is frozen, and ReLU6 needs only a 1-2-bit mask. But the GroupNorm inputs (172 MiB) are needed by any engine, because the input-gradient of GroupNorm depends on the normalized input. So in measured memory, which kind of parameter is trained (bias, lite residual or weight) moves almost nothing: LB, B and Full are 537-599 MiB. **The earliest layer that receives a gradient sets the memory, and the batch size multiplies it.** These are the only two levers this proposal treats as real.

---

## 1. Research question and why it matters

**Question.** With only k = 2-10 labeled images per class, what is the shallowest backprop cutoff (adapting only TinyTL's last stages) whose accuracy stays within 2 pp of full-depth TinyTL-L+B, and how does that cutoff compare with TinyTL's other memory lever, training at batch 1?

**Device scenario, tied to the evidence.**
- TinyTrain's target is a Pi Zero 2-class board with 512 MB of RAM. TinyTrain says TinyTL's 541-587 MB does not fit, but those figures are for batch 100.
- At TinyTL's own published batch 8, our measured training step for TinyTL-L+B already needs **557.5 MiB (585 MB)**, which is more than the whole board before the OS and the app. The paper's 37 MB comes from the profiler and is not realized in PyTorch; measured FT-Full is only 1.07x LB.
- A practitioner who has to fit about 128-256 MiB therefore has exactly two levers, both measured:
  - cut the depth: T3 at 169 MiB is 3.3x less than LB, T4 at 105 MiB is 5.3x less, FT-Last at 67 MiB is 8.3x less;
  - cut the batch: LB at batch 1 is 123 MiB, 4.5x less.
  By contrast, freezing every lite residual (TinyTL-B) saves only 20.5 MiB (3.7%).
- On-device users label very few images. The two papers disagree on what depth buys when labels are few:
  - TinyTL on full Flowers (20 per class): L+B 95.5 vs FT-Last 90.1, a 5.4 pp gap.
  - TinyTrain few-shot on Flowers (ProxylessNAS, Table 1): TinyTL 86.2 vs LastLayer 85.5, a 0.7 pp gap.
  - TinyTrain Appendix F.1 says the important layers lie among the last few, and that inspecting 30-44% of layers suffices. T3 (blocks 13-21, 41% of blocks) is in that range.

  So the answer is label-dependent and nobody has measured it inside TinyTL. This experiment does that.

---

## 2. Story: kept, changed, dropped

| story element | decision | reason (evidence) |
|---|---|---|
| ①② On-device motivation; activations dominate training memory | **Kept** | Confirmed by measurement: of LB's 557.5 MiB, 511.8 are tensors autograd saves for backward, and weights are 18.2. |
| ③ TinyTL mechanism (frozen weights, bias + lite residual) as the reference method; our reproduction 95.20 vs 95.5 | **Kept**, wording fixed | One seed and the upstream protocol (test split used for checkpoint selection). Report it as a sanity reproduction, not as a confirmed match. |
| ③ "37 MB vs 391 MB" memory narrative | **Changed** | That is the profiler. Measured: 557.5 vs 598.7 MiB at batch 8 (1.07x) and 122.6 vs 160.0 at batch 1 (1.30x). TinyTL-B is 537 MiB measured, not 32. Keep the profiler numbers in a separate "paper's model" column and explain the gap with the saved-tensor breakdown (§0). Every saving we claim comes from Table M. |
| ④ Data scarcity as the second obstacle; k-shot protocol | **Kept** | Already implemented: k per class drawn from the original train split, the same images for every arm at a given seed. |
| ⑤ "TinyTL clearly beats last-layer in TinyTrain's few-shot results" | **Changed** | True only for the nine-domain average (63.7 vs 57.0). On Flowers it is 86.2 vs 85.5. That small gap is the motivation for H1 and H3. |
| ⑤ "Important layers are in the later part" | **Kept, re-sourced** | It is in TinyTrain Appendix F.1 (and Figs. 7/8), not the main text. It becomes the prediction tested by H1. |
| ⑤ TinyTL "catastrophically degrades with small batch" (TinyTrain) vs "GN works at batch 1" (TinyTL) | **Kept as H2, reframed** | Batch 1 is also a measured memory lever (557.5 to 122.6 MiB), so the contradiction now answers a device question too. We can only test the GN checkpoint; if GN works, the contradiction must come from TinyTrain's setup (BN or meta-trained backbones, lite-residual-only variant), which we do not test. |
| ⑦ Task-adaptive (Fisher) selection of which blocks train lite residuals | **Changed to an exhaustive depth-cutoff sweep** | (1) Memory is set by the earliest selected block, so a non-contiguous selection costs as much as truncating at its earliest member. Choosing which lite residuals to train while biases still backprop to the stem saves at most about 21 MiB (under 4%). (2) With one dataset there is one task, so "task-adaptive" cannot be tested; per-seed differences in a selection would be sampling noise. (3) The cutoff is one-dimensional with 6 levels, so sweeping it costs little and gives an upper bound on what any selection rule can achieve on this axis. The Fisher computation survives only as an optional zero-training diagnostic (O3). |
| ⑦ Imprinting head initialization | **Dropped from core** (optional O5, k=2 only) | It does not bear on memory, and no evidence suggests it matters. At k=5 train top-1 is 99-100% by epoch 10-15, so under a 50-epoch budget the head init is probably washed out. A 1 pp effect would need about 18 seeds. |
| ⑧ "Same memory budget, fixed vs task-adaptive, each with/without imprinting" | **Replaced** | A memory-accuracy Pareto over cutoffs plus one batch-1 point, with paired non-inferiority tests against LB. |
| ⑧ Batch-size study across several k | **Narrowed** to k=5 | Batch-1 runs cost 4x as much (about 31 min each). One k with a decisive "catastrophic or not" rule is affordable; more k is not. |
| Checkpoint chosen on the 1020-image val split | **Changed to final epoch** | A device with 5 labels per class has no 1020-image val set. Learning curves never decline (seed 0: val 86.2 at epoch 15, 87.3 at epoch 50). Val agrees with test poorly on small differences (r = 0.169 within r1/r2) and picked the wrong config in 1 of 3 seeds. Val is used only for the one-time learning-rate pilot and for diagnostics. |
| Resolution / pooling-ratio axis (previous experiment) | **Dropped** | Closed: r4 loses 7.9 pp for 12 MiB, and r1 gains +1.2 pp (not significant) for +50 MiB. |
| Risk: Flowers only | **Kept**: conclusions limited to near-domain fine-grained; Aircraft is optional O6 | Flowers is where TinyTrain shows the smallest TinyTL-vs-LastLayer gap, so a "depth does not matter" result here may not transfer. |

Revised one-line story: *TinyTL's memory saving is a property of its analytical model. In measured PyTorch memory, the cost of on-device transfer is set by how deep the gradient goes and by the batch size. We measure how deep few-shot adaptation actually needs to go, and whether batch 1 is a better way to fit a 512 MB-class device.*

---

## 3. Hypotheses and decision rules

**Common definitions.**
- Metric: top-1 (%) on the 6,149-image Flowers102 test split, of the final-epoch (epoch 50) model.
- Unit: seed. A seed fixes the k-shot subset, the head init and the data order. At a given (k, seed) every arm sees the same images.
- d_X = acc(X) − acc(LB-b8), paired by seed.
- [L, U] = two-sided 90% t-interval of mean d_X (equivalent to one-sided 95% bounds).
- Margin δ = 2.0 pp. This is the smallest method gap the TinyTL paper treats as decisive (TinyTL-B 93.5 vs L+B 95.5). A loss smaller than that would not change the paper's own recommendation.

**Category labels** (used for H1 and for the secondary checks at k=2 and k=10):
- *retains*: L > −2.0. If also U < 0, call it *retains, small real cost*.
- *loses more than the margin*: U < 0 and L ≤ −2.0.
- *inconclusive*: L ≤ −2.0 and U ≥ 0. This triggers one 5-seed extension (§6); after that, report the CI as a bound.
- *truncation helps* (the story's overfitting premise): L > 0.

**Resolution with the planned seeds.** Planning SD is the measured paired SD of 1.83 pp at k=5, and 2.5 pp as a pessimistic case. From `design\power_calc.py`:

| SD | n | 90% CI half-width | P(retains given true d = 0) | P(retains given true d = −1) | P(retains given true d = −3) | inconclusive zone of true d |
|---|---|---|---|---|---|---|
| 1.8 | 10 | 1.04 | 0.95 | 0.47 | <0.001 | [−1.04, −0.96], negligible |
| 2.5 | 10 | 1.45 | 0.76 | 0.28 | 0.001 | [−1.45, −0.55] |
| 2.5 | 15 | 1.14 | 0.91 | 0.42 | <0.001 | [−1.14, −0.86] |

With 10 seeds at SD 1.8, essentially every outcome is labeled. If SD turns out to be 2.5, the extension to 15 seeds shrinks the unresolved zone to 0.3 pp.

### H1 (primary): truncated TinyTL retains full-depth accuracy at k=5

Prediction (TinyTrain F.1): T3 retains the accuracy.

Procedure: k=5, n=10 (seeds 0-9). Fixed-sequence non-inferiority test **T3 → T4 → FT-Last**, ordered by decreasing memory. Each step is a one-sided test at α = 0.05. Stop at the first arm that does not *retain*. This controls family-wise error at 5% without splitting α.

Answer: s* is the last arm that passed. Its measured memory saving vs LB comes from Table M (3.3x, 5.3x or 8.3x).

Conditional arms:
- If T3 fails, run **T2** (226 MiB, 2.5x) with the same 10 seeds and test it.
- If T4 passes but FT-Last fails, run **T5** (68.4 MiB, only 1 MiB above FT-Last) to test whether the memory floor can still carry stage-5 adaptation.

All arms' CIs are reported regardless of where the sequence stops. *Truncation helps* is checked for each arm with a Holm correction across the arms that were run.

### H2 (secondary): GN TinyTL-L+B does not degrade catastrophically at batch 1 (k=5)

Procedure: d_b1 = acc(LB-b1) − acc(LB-b8), paired over seeds 0-5 (n=6). Labels:
- *no meaningful degradation*: L > −2. TinyTL's claim holds in few-shot.
- *catastrophic*: U < −5. TinyTrain's claim is reproduced with GN.
- *moderate degradation*: U < 0, L ≤ −2, U ≥ −5.
- *inconclusive*: L ≤ −2 and U ≥ 0. Extend to n=10 once.

Power at n=6:
- if the true drop is 8 pp, P(U < −5) = 0.98 at SD 1.8 and 0.82 at SD 2.5;
- if the true drop is 0, P(L > −2) = 0.76 at SD 1.8 and 0.48 at SD 2.5. The extension exists for this case.

### H2b (derived, no extra runs): at a ≤128 MiB budget, which lever wins?

Comparison: e = acc(LB-b1, 122.6 MiB) − acc(T4-b8, 105.1 MiB), paired over seeds 0-5. Both fit 128 MiB.
- "batch lever better" if L > 0.
- "depth lever better" if U < 0.
- Otherwise "no detectable difference". Then recommend T4: it uses 14% less measured memory, and on our GPU batch 1 costs 8x more GPU time per image (58.6 vs 7.2 ms/image measured for LB).

At larger budgets the comparison is read off the Pareto plot. LB-b1 at 123 MiB either dominates T3-b8 at 169 MiB or it does not; that is the same paired test with T3.

### H3 (secondary): the benefit of depth shrinks as labels get scarcer

Definitions: g_X(k) = mean over seeds of [acc(LB, k) − acc(X, k)], for X ∈ {FT-Last, T3}.

Primary test: X = FT-Last. Welch t-test on per-seed paired differences, k=5 (n=10) vs k=2 (n=10), with the 90% CI of g(5) − g(2):
- "depth matters less with fewer labels" if L > 0. This would reconcile TinyTL's 5.4 pp with TinyTrain's 0.7 pp.
- "depth matters more with fewer labels" if U < 0.
- "no large trend" if the CI lies inside (−2.5, +2.5).
- Otherwise inconclusive; report the CI.

MDE at 80% power: 2.3 pp (SD 2.0 at both k) to 2.6 pp (SD 2.5 at k=2).

k=10 (n=3) is descriptive only (90% CI half-width ±1.7 pp at SD 1.0). It checks monotonicity (g(2) < g(5) < g(10) as point estimates) and connects to the paper's full-data 5.4 pp. The same analysis is repeated for X = T3 as a secondary.

---

## 4. Conditions

All arms use the same network: ProxylessNAS-Mobile + lite residuals (r=2, official insertion), GN + weight standardization, the `ws+gn` checkpoint, and a new 102-way head (dropout 0.2). Frozen weights get 8-bit k-means quantization (upstream rule: every frozen conv or linear weight, so truncated arms also quantize their frozen early lite residuals). Arms differ only in which parameters receive gradients (§0, Table M).

| block | arm | trained | batch | LR | k | seeds | runs |
|---|---|---|---|---|---|---|---|
| P (pilot) | LB, T3, FT-Last | as Table M | 8 | 2e-4, 8e-4 (+3.2e-3 if 8e-4 wins by ≥1 pp) | 5 | 100, 101 | 12-18 |
| A (core) | LB-b8 | TinyTL-L+B | 8 | LR_LB | 5 | 0-9 | 10 |
| A | T3-b8 | stages 3-5 | 8 | LR_T3 | 5 | 0-9 | 10 |
| A | T4-b8 | stages 4-5 | 8 | LR_T4 | 5 | 0-9 | 10 |
| A | FT-Last-b8 | classifier | 8 | LR_Last | 5 | 0-9 | 10 |
| C (core) | LB-b1 | TinyTL-L+B | 1 | 1.5 × LR_LB | 5 | 0-5 | 6 |
| B2 (core) | LB-b8, T3-b8, FT-Last-b8 | as above | 8 | as chosen | 2 | 0-9 | 30 |
| B10 (core) | LB-b8, T3-b8, FT-Last-b8 | as above | 8 | as chosen | 10 | 0-2 | 9 |
| A-T2 (cond.) | T2-b8 | stages 2-5 | 8 | LR_LB if LB and T3 agree, else 2-run pilot | 5 | 0-9 | 10 |
| A-T5 (cond.) | T5-b8 | stage 5 | 8 | LR_Last if T3 and FT-Last agree, else pilot | 5 | 0-9 | 10 |
| B2-T4 (cond.) | T4-b8 | stages 4-5 | 8 | LR_T4 | 2 | 0-9 | 10 |
| A-ext / C-ext (cond.) | arms in an inconclusive comparison, plus LB | | | | 5 | 10-14 / 6-9 | ≤15 / 4 |

Notes:
- T4 takes T3's LR if T3 and FT-Last chose the same one; otherwise it gets a 4-run pilot.
- LB-b1's factor of 1.5 keeps upstream's batch-1/batch-8 ratio (3e-4 / 2e-4). If LR_LB stays at 2e-4, LB-b1 runs exactly the published batch-1 recipe.
- k=10 uses the whole train split, so its seeds vary only init, order and augmentation.

---

## 5. Protocol

**Data pool and sampling.**
- Flowers102 original splits from `setid.mat`: train 1,020 (10 per class), val 1,020, test 6,149.
- For each (k, seed), `build_root(k, seed)` (unchanged from `fewshot_resolution.py`) draws k images per class from the train split with `random.Random(seed)`. The same subset is shared by every arm, which makes the design paired.
- k ∈ {2, 5, 10}, i.e. 204 / 510 / 1,020 training images.
- Main seeds: 0-9. Pilot seeds: 100-101. Test images are never in any pool; val images are never trained on.

**Model selection.** None. The epoch-50 model is evaluated on test exactly once per run. Val top-1 is logged every 5 epochs (10 passes, about 30 s per run) for learning curves only.

Protocol bridge (before block A, about 2 min of GPU): test-evaluate the saved final-epoch `checkpoint.pth.tar` of the existing r2 k=5 runs (seeds 0-2). Compare with their best-val test numbers (83.54 / 83.70 / 81.72) to document the switch. The bridge does not decide anything.

**Learning rate.** One pilot at k=5 on val; the chosen values are frozen in `lr_choice.json` before any main run.
- For each of LB, T3 and FT-Last: run 2e-4 (official) and 8e-4 on seeds 100-101. Score is mean final-epoch val top-1.
- If 8e-4 beats 2e-4 by ≥ 1.0 pp, also run 3.2e-3.
- Take the best LR. Ties within 1.0 pp go to the lower LR, which favors the published recipe.
- Inheritance rules for T4, T2, T5 and LB-b1 are given in §4.
- Rationale: run.py's single 2e-4 for every method was tuned for L+B on full data with test-set selection. Tuning each arm on the same grid keeps the comparison fair, and val is reliable for the multi-point differences an LR change causes, even though it is unreliable for 1 pp config differences.
- The same LRs are used at every k. This is a stated limitation, with a check: at the final epoch, train top-1 must be ≥ 95% and val must rise less than 1 pp from epoch 40 to 50. An arm that fails is flagged as under-trained.

**Training budget.** The official recipe, unchanged except for LR and batch:
- 50 epochs, cosine schedule, Adam, label smoothing 0.7, no weight decay, `resize_scale` 0.22, `distort_color` None, 224 px, `n_worker` 0.
- Steps scale with k: batch 8 gives 1,300 / 3,200 / 6,400 steps at k = 2 / 5 / 10; LB-b1 at k=5 takes 25,500 steps.

**Evaluation.** Each run writes to `result.json`:
- test top-1 (primary), top-5 and loss, upstream center-crop transform;
- final-epoch val top-1, the val curve and final-epoch train top-1;
- wall-clock training seconds, steps, trainable parameter count and the profiler MB.

The analysis uses only paired differences.

**Memory.** Taken from the harness (`measure_memory.py`, i.e. `mem_harness.py` moved into the project and sharing `set_trainable` with the trainer), never from training runs. A training run's own peak includes evaluation at batch 64; for example the old r2 runs logged 686 MiB.
- Protocol: synthetic input; 2 warm-up steps, then 3 measured steps; one throw-away arm first in each process so the one-time ~16 MiB workspace is excluded; three separate processes in different arm orders.
- Report the maximum and the range per arm, at batch 8 and 1. Report the saved-tensor breakdown and the profiler number alongside, as explanation only.

**Time.** `measure_time.py` measures GPU ms per train step (synthetic) for every arm at batch 1 and 8, after 10 warm-up steps and over 50 timed steps. **It must run with no other GPU application open.** Today the GPU was at 40% utilization from another program, which is why I did not time anything.
- Compute savings are claimed only as measured step-time ratios (for example 57.4 → 40.3 ms/step, −30%, for T3 vs LB at batch 8).
- Per-run wall-clock is reported only as "on our machine", because data loading dominates with `n_worker` 0.

---

## 6. Run list, GPU-hours, order and gates

**Run-time model** (context.md, ±30%):
- 16.8 ms per training image at batch 8 and 68 ms at batch 1, plus 3 ms per eval image.
- Plus an assumed 1.0 min of per-run setup (model build, k-means quantization, profiler). This should be measured on the first pilot run.
- Val every 5 epochs adds 0.52 min; one test pass adds 0.31 min.

Resulting per-run cost:

| run type | minutes |
|---|---|
| k=2, batch 8 | 4.7 |
| k=5, batch 8 | 9.0 |
| k=10, batch 8 | 16.1 |
| k=5, batch 1 | 30.7 |

Truncated and FT-Last runs spend less GPU time per image, so they are probably 10-25% faster; the estimates use the LB rate throughout as a conservative bound.

| order | block | runs | GPU-h | status / gate |
|---|---|---|---|---|
| 0 | M: memory harness ×3 processes, timing harness, protocol bridge | n/a | 0.15 | core. Gate G0: harness trainable counts must equal Table M; smoke run (1 epoch, k=2) of each arm × batch. |
| 1 | P: LR pilot | 12-18 (+4 if T4 needs one) | 1.8-3.3 | core. Gate G0b: if LB at 2e-4 gives final val < 84% (old runs 85.7-87.3), stop and debug before block A. Freeze `lr_choice.json`. |
| 2 | A: k=5 × {LB, T3, T4, FT-Last} × 10 seeds, **seed-major order** (all arms of seed 0, then seed 1, …) so that a partial block is still paired | 40 | 6.0 | core. **Gate G1**: run the H1 analysis. Conditionals: A-T2, A-T5, A-ext. |
| 3 | C: LB-b1 × seeds 0-5 | 6 | 3.1 | core. **Gate G2**: H2 inconclusive → C-ext (+4 seeds). |
| 4 | B2: k=2 × {LB, T3, FT-Last} × 10 | 30 | 2.35 | core. If T4 passed at k=5, add B2-T4. |
| 5 | B10: k=10 × {LB, T3, FT-Last} × 3 | 9 | 2.4 | core, last; cut to 2 seeds if behind schedule (descriptive only). |
| | **core total** | 97-107 | **15.8-17.3** (realistic about 15; +30% pessimistic about 21) | |
| c1 | A-T2 (only if T3 fails H1) | 10 | 1.5 | conditional |
| c2 | A-T5 (only if T4 passes and FT-Last fails) | 10 | 1.5 | conditional (exclusive with c1 in practice) |
| c3 | A-ext (only if an H1 comparison is inconclusive): +5 seeds for LB and ≤ 2 arms | ≤15 | ≤2.3 | conditional |
| c4 | C-ext (only if H2 is inconclusive) | 4 | 2.0 | conditional |
| c5 | B2-T4 | 10 | 0.8 | conditional |
| | **conditional max** | | **≈ 7** | |
| O1 | FT-Full at k=5: LR pilot {5e-5, 2e-4} × 2 seeds + 10 seeds. Tests whether TinyTL beats plain fine-tuning when measured memory is only 7% lower. | 14 | 2.2 | optional, rank 1 |
| O2 | Paper-protocol reproduction of the 5 baselines (run.py, train+val 20 per class, upstream selection, seed 0) | 5 | 2.9 | optional, rank 2 |
| O3 | Fisher-potential diagnostic (TinyTrain Eq. 2) per stage, from each seed's k=5 samples: does it rank late stages highest, and would a Fisher-under-budget rule pick s*? | 0 training | 0.1 | optional, rank 3 |
| O4 | Combined levers T3-b1 and T4-b1 at k=5, seeds 0-5 (67 / 52 MiB) | 12 | ≈ 4.4 | optional, rank 4 |
| O5 | Imprinting at k=2 for FT-Last and LB, 10 seeds | 20 | 1.6 | optional, rank 5 |
| O6 | Aircraft k=5 {LB, T3, FT-Last} × 5 seeds + 1-seed LR pilot (after a few-GB download) | ≈ 21 | ≈ 3.3 | optional, rank 6 |
| | **all-in maximum** | | **≈ 17.3 + 7 + 14.5 ≈ 39** | within 40 |

Suggested schedule. P must be analyzed and `lr_choice.json` frozen before A starts, so P runs in the daytime:
- Day 1: M and P (2-3.3 h), then fix the LRs.
- Night 1: A and C (about 9.1 h).
- Day 2: G1/G2 analysis.
- Night 2: B2, B10 and whichever conditionals were triggered (4.8 h plus up to about 7 h).
- Optionals only after the core write-up tables exist.

---

## 7. Implementation plan (about 1.5-2 person-days of coding; nothing in the existing files is modified)

| # | file (new unless noted) | what | lines |
|---|---|---|---|
| 1 | `fewshot_depth.py` (derived from `fewshot_resolution.py`) | Keep `splits`, `link`, `build_root` and `test_root` unchanged, so sampling is identical to the old runs. Replace the ratio insertion with the official `LiteResidualModule.insert_lite_residual(net, 2, 'bilinear', 1.0, 5, 'relu', 2)`. Add `block_index(name)` and `set_trainable(net, arm)` (rule exactly as in `design\mem_harness.py: build`; arms `last`, `T5`, `T4`, `T3`, `T2`, `LB`, `full`), replacing lines 164-170; assert the trainable count against Table M. New args: `--arm --batch --lr --val_every 5 --k --seed --epochs 50`. Config uses `train_batch_size=args.batch`, `init_lr=args.lr`, `validation_frequency=args.val_every`. After `run_manager.train`, delete the `model_best` reload (lines 196-197) and test the in-memory final model. Parse the val curve and the final train top-1 from `valid_console.txt` into `result.json`. Rename the run peak to `run_peak_incl_eval_MiB` so nobody cites it as train memory. Optionally save a checkpoint only at the last epoch. Output directory: `runs/depth-k{k}-{arm}-b{batch}-lr{lr}-seed{seed}`. | ≈ 240 total, ≈ 80 new/changed |
| 2 | `measure_memory.py` | Move `design\mem_harness.py`; import `set_trainable` from #1 so there is one definition; add a throw-away warm-up arm per process; driver that runs 3 processes and writes `memory_table.json` (max and range). | ≈ 140 |
| 3 | `measure_time.py` | Extend `time_steps.py` to all arms × {1, 8}; log `nvidia-smi` utilization before starting and refuse to run if it is above 5%. | ≈ 45 |
| 4 | `make_runlist.py` + `run-depth-sweep.sh` | Generate `runlist.csv` (`block,k,arm,batch,lr,seed`) in the pre-registered seed-major order from `lr_choice.json`. Bash driver modeled on `run-fewshot-sweep.sh` (the PowerShell driver was killed twice): skip runs whose `result.json` exists, stop on the first failure. Remind the team to disable sleep (`powercfg`, as in the earlier report). | ≈ 40 + 40 |
| 5 | `eval_final_ckpt.py` | Protocol bridge: load `checkpoint.pth.tar` of old r2 seeds 0-2 and run the test pass. | ≈ 40 |
| 6 | `analyze_depth.py` | Load `result.json` files; paired differences vs LB-b8 per (k, seed); 90% t-CIs; fixed-sequence H1 with category labels; H2 and H2b labels; Welch H3; Holm for "truncation helps"; LR pilot table; Pareto plot (test top-1 vs measured MiB, batch-8 and batch-1 markers, profiler MB as a second x-axis); learning-curve panel. | ≈ 180 |
| 7 (O3) | `fisher_stage.py` | Hooks on block outputs over the k-shot set: Δo per channel = (1/2N) Σ_n (Σ_d a·g)², summed per stage and divided by parameters and MACs, followed by TinyTrain's greedy rule under a measured-MiB budget. | ≈ 70 |
| 8 (O5) | `--imprint` in #1 | Extract un-augmented features of the k-shot set; set the classifier weights to the L2-normalized class means × scale; bias 0. | ≈ 25 |

Suggested split:
- Person 1: #1 and #5.
- Person 2: #2, #3 and #4.
- Person 3: #6 and the report skeleton (Table M, the §0 explanation, the profiler-vs-measured section).

Pre-launch checks:
- trainable counts match Table M;
- a 1-epoch smoke run passes for every (arm, batch);
- the first pilot run's setup time is measured and §6 updated.

---

## 8. Risks, fallbacks and expected outcomes

| risk | fallback |
|---|---|
| Paired SD above 1.8 (different method families may correlate less across seeds) | Use the pre-registered extension (+5 seeds for inconclusive comparisons only, once). After that, report the CI as a bound and do not add more runs. |
| LR pilot picks badly (2 seeds, noisy) | The 1 pp tie rule favors the official recipe. If an H1 conclusion hinges on an arm whose pilot margin was < 1.5 pp, run 3 seeds of that arm at the runner-up LR as a sensitivity check (≤ 0.5 h, from the conditional budget). |
| Run-time model off by +30% | Block order is priority order. B10 is last and descriptive, and can drop to 2 seeds; optionals are cut first. |
| Batch-1 runs slower than 40 min | Keep n=6 and report only "catastrophic or not" (the well-powered part of H2). |
| Machine sleep, crash, or PowerShell killing jobs | Bash driver with skip-if-done (the existing workaround that works); `powercfg` standby 0. |
| GPU shared with other apps | Memory measurements are unaffected (per process). Timing is invalid unless the GPU is idle, enforced in #3. |
| Under-training at k=2 (1,300 steps) | Train ≥ 95% / val-plateau check. If an arm fails, flag it, and if the budget allows run a 2-seed k=2 LR check for that arm. |
| Generality (one near-domain dataset) | Limit claims to Flowers102 and fine-grained near-domain; O6 Aircraft if time allows. |
| Measured memory is PyTorch-specific | State every claim as "in PyTorch fp32". The saved-tensor breakdown shows which part is framework (conv inputs, fp32 masks) and which is intrinsic (GN inputs). Profiler numbers stay in a separate column. |
| Quantization asymmetry (truncated arms also quantize frozen early lite residuals) | 8-bit k-means error is negligible and this is upstream's rule; note it. |
| H2 tests only GN | Say so. If GN works at batch 1, the TinyTrain claim is attributed to an untested difference in their setup, not "refuted". |

**Expected outcome → conclusion**

| result | conclusion the team writes |
|---|---|
| H1: T3, T4 and FT-Last all *retain* | At k=5, adapting beyond the classifier buys < 2 pp: FT-Last (67 MiB, 8.3x less than TinyTL's 557 MiB) is the device choice. TinyTL's adaptation only pays with more labels (read with H3). |
| H1: T3 and T4 retain, FT-Last loses | Adapting the last two stages keeps TinyTL accuracy at 105 MiB (5.3x less). Deeper backprop is wasted memory in few-shot. The c2 (T5) result says whether the 68 MiB floor can carry stage 5. |
| H1: T3 retains, T4 loses | Three stages needed: 169 MiB (3.3x). Consistent with TinyTrain's "30-44% of layers". |
| H1: T3 loses, T2 retains | Few-shot TinyTL needs stages 2-5; the measured saving is at most 2.5x (226 MiB). |
| H1: T3 and T2 lose | Early-stage adaptation matters even at k=5. Full-depth TinyTL is justified, and the only measured memory lever left is the batch (H2). |
| Any arm with L > 0 | Overfitting premise supported: full-depth adaptation hurts at this k. |
| H2: no meaningful degradation (L > −2) | GN TinyTL trains at batch 1 in few-shot within 2 pp at 123 MiB (4.5x less). TinyTrain's "catastrophic" claim does not hold for TinyTL's own GN checkpoint; the cause lies in their setup (untested). |
| H2: catastrophic (U < −5) | Even with GN, batch 1 breaks few-shot TinyTL under its published recipe. TinyTrain's observation is reproduced, so the batch lever is unusable and depth is the only option. |
| H2: moderate | Batch 1 costs 2-5 pp. It is usable only where depth truncation costs more (decided by H2b). |
| H2b: batch better / depth better / tie | Under ≤ 128 MiB, recommend LB-b1, T4-b8, or T4-b8 respectively (a tie goes to T4 for less memory and less GPU time). |
| H3: g(5) − g(2) > 0 | TinyTL's advantage over FT-Last shrinks with fewer labels. This reconciles TinyTL (5.4 pp at full data) with TinyTrain (0.7 pp few-shot on Flowers). |
| H3: no large trend | The depth benefit is roughly constant over k = 2-5 (bounded by the CI). The paper-to-TinyTrain difference must come from their protocol (meta-training, various-way tasks). |
| H3: g(5) − g(2) < 0 | Depth matters more with fewer labels, against the overfitting premise. Truncation is riskier at small k. |
