# Proposal r2: How deep must few-shot TinyTL back-propagate, and when is batch 1 the better way to fit memory?

Flowers102 (near-domain) and FGVC-Aircraft (far-domain), k = 5, ProxylessNAS-Mobile with TinyTL's `ws+gn` lite-residual checkpoint, RTX 3060.

New zero-training measurements for this round. All files are in `design\`; none took longer than 3 minutes.

| file | what it measures | result file |
|---|---|---|
| `engine_bound.py` | three memory accountings, plus GN replaced by a frozen affine | `engine_bound_result.json` |
| `mem_repeat.py` | repeat of the memory harness, with nvidia-smi utilization logged | `mem_repeat_result.json` |
| `macs.py` | analytic backward MACs | `macs_result.json` |
| `decode_time.py` | CPU cost per image of Flowers vs Aircraft-sized JPEGs | `decode_time_result.json` |
| `power_r2.py`, `power_r2b.py` | Monte Carlo of the r2 decision rules, including the priors and Holm | `power_r2_result.txt`, `power_r2b_result.txt` |

The GPU was shared with another application (11-36% utilization). This does not affect per-process allocator memory, and nothing was timed on the GPU.

## Changes from r1

| # | r1 | r2 | trigger |
|---|---|---|---|
| 1 | Flowers only; claimed "nobody has measured it inside TinyTL" | **Flowers + Aircraft at k=5.** AdapterDrop (TinyTrain Table 6) is cited as the closest prior work, and its Flower/Aircraft numbers are the pre-registered priors. The contribution is restated against it. | B1 (a) |
| 2 | 3 labels, 90% CIs, memory-ordered fixed sequence | 4 labels, 93% CIs at every look, Holm over {T3, T4, FT-Last} per dataset. Simulated label probabilities replace the "zone" column. The all-arm extension is budgeted. | B2 |
| 3 | "TinyTL's saving is a property of its analytical model" | Three accountings per arm (profiler / ideal-engine lower bound / measured PyTorch), plus a GN → frozen-affine ablation. Depth and batch are real levers in both non-profiler accountings; the method-type lever is 10.5× / 1.9× / 1.07×. **The profiler is right only for frozen-statistics norms.** | B3, S10 |
| 4 | LB-b1 at 1.5 × the pilot LR, on Flowers | LB-b1 at the **published batch-1 LR for the dataset**, moved to **Aircraft**, where depth is expensive and H2b decides something. A negative H2 label is only reported after a pre-registered LR check. | B4 |
| 5 | k=2 block (H3) with a vacuous convergence check | H3 moved to optional O2, which now includes the step-matched control. The core gets a convergence check that can fail (half-budget val gain), with a conditional 100-epoch test. | B5 |
| 6 | Fisher optional; "exhaustive sweep / upper bound" wording | Fisher-under-budget diagnostic is **core**, computed with a ProtoNet-equivalent imprinted head and pre-registered as descriptive. The "exhaustive" and "upper bound" wording is deleted. T2 is core on Aircraft (as the follow-up arm) and stays conditional on Flowers (rebuttal in B6). | B6 |
| 7 | B10 (k=10) core | Moved to optional O7. | budget |
| 8 | Compute claims from GPU ms ratios | Analytic backward MACs per arm. The H2b tie-break uses memory and MACs. | S1 |
| 9 | RNG pairing, trainer details, disk, provenance | All adopted (§7). | S3-S6 |

Core budget (realistic, arm-specific rates): **16.3 GPU-h, plus 3.6 h of likely conditionals = 19.8 h.** Conservative (every arm at LB's rate): 21.8 h. Cut relative to the reviewer's plan: k=2/H3 and the Aircraft seed extension, both now optional (O2, O1). §6 gives a pre-registered cut order in case the run-time model proves optimistic.

---

## Responses to the blocking issues

### B1. Accept, option (a)

TinyTrain Table 6 (`tinytrain_paper.txt` lines 1233-1239, ProxylessNAS rows) already measured truncated TinyTL ("AdapterDrop"). r1's novelty sentence is withdrawn.

**What this study adds over Table 6:**
1. **Memory of truncation.** Table 6 reports none for AdapterDrop. We give the profiler, an ideal-engine bound and measured PyTorch memory.
2. **TinyTL's own recipe.** We use TinyTL's released ImageNet-pretrained lite-residual checkpoint with its L+B recipe. Table 6 uses meta-trained backbones with fresh, lite-residual-only adapters, about 25 samples, 40 iterations and batch 100.
3. **Paired inference.** Seed-paired tests against a pre-registered 2 pp margin, with Table 6 as the prior. This makes it a replication of the depth ladder under a different protocol.
4. **The batch lever** compared with depth at matched measured memory.
5. **The GN mechanism** behind the profiler gap (B3).

**Pre-registered priors.** Our arms map onto the AdapterDrop rungs by the share of adapter blocks kept (lite residuals sit on blocks 1-21):
- T2 keeps 13/21 blocks (62%), between AD-25% and AD-50%;
- T3 keeps 9/21 (43%), about AD-50%;
- T4 keeps 5/21 (24%), about AD-75%;
- FT-Last corresponds to LastLayer.

| arm | analogue | Flower prior d (pp) | Aircraft prior d (pp) |
|---|---|---|---|
| T2 | AD-25% to AD-50% | −0.6 to −1.0 | −2.1 to −3.1 |
| T3 | AD-50% | −1.0 | −3.1 |
| T4 | AD-75% | −1.9 | −10.6 |
| FT-Last | LastLayer | −0.7 | −17.6 |

d is the arm minus TinyTL (86.2 Flower, 70.3 Aircraft).

Expected labels under these priors (`power_r2b.py`, Holm, 93% CIs, extension on Flowers):

| dataset (paired SD) | T3 | T4 | FT-Last |
|---|---|---|---|
| Flowers (1.8) | retains 32% (+22% unadjusted only); significant loss, size unresolved 43% | size unresolved 91% | retains 47% (+27% unadjusted only); size unresolved 23% |
| Aircraft (2.5) | size unresolved 79%; inferior by more than the margin 17% | inferior about 100% | inferior about 100% |

So the two datasets are predicted to land at opposite ends. If they do not, that is itself a finding about the protocol differences listed above.

**Aircraft data prep** is planned in §5 and budgeted in §6.

### B2. Accept

**Labels.** L and U are the bounds of the two-sided 93% t-CI of the mean paired difference d (one-sided 3.5%):

| label | rule |
|---|---|
| *retains* | L > −2 |
| *inferior by more than the margin* | U < −2 |
| *significant loss, size unresolved* | U < 0, L ≤ −2, U ≥ −2 |
| *inconclusive* | L ≤ −2, U ≥ 0 |

Only "inferior by more than the margin" can trigger a "deeper adaptation is needed" conclusion. "Size unresolved" is reported as "costs between 0 and |L| pp". The H2 labels are defined the same way, with *moderate* meaning −5 ≤ U < −2 (§3).

**Multiplicity.** The r1 fixed sequence is replaced by **Holm over {T3, T4, FT-Last} within each dataset**, testing NI p-values at one-sided 0.035. The r1 sequence assumed accuracy falls monotonically as the cutoff gets shallower. The Flower prior is not monotone (LastLayer 85.5 > AD-75% 84.3), so a memory-ordered sequence could stop at T4 and never test FT-Last. T2 and T5 (conditional) are labeled *exploratory*, outside the error-rate claim.

**Error rates** (`power_r2.py`, sample SD varies, one extension look for inconclusive arms only). Type-I rate of "retains" at true d = −2:

| design | SD 1.8 | SD 2.5 | SD 3.0 |
|---|---|---|---|
| n 10 → 15 | 4.2% | 5.0% | 5.3% |
| n 5 → 10 | 5.3% | 5.7% | 5.9% |

We claim "about 5%", not "controls at 5%". The probability that the extension fires is in the label tables of §3. The worst case, where every Flowers arm needs the extension, is in the conditional maximum (§6).

### B3. Accept

The three accountings come from `engine_bound.py`. The engine bound is defined as follows:
- autograd-saved tensors in full fp32, if any saver is a norm layer or a conv/linear with a trainable weight;
- 1 bit per element, if the only savers are ReLU/ReLU6;
- 0, if the only savers are frozen convs or shape-only ops;
- plus fp32 weights, grads and Adam m and v for trainable parameters, and the input batch.

Transients are excluded; the forward peak adds at most about 26 MiB at batch 8.

**Table M2. Memory per arm (MiB; profiler in MB)**

| arm | b8 profiler | b8 engine bound | b8 measured | b1 profiler | b1 engine bound | b1 measured |
|---|---|---|---|---|---|---|
| FT-Last | 31.6 | 24.2 | 67.4 | 8.5 | 20.2 | 25.1-25.9 |
| T4 | 34.6 | 63.4 | 105.0-105.2 | 11.4 | 41.9 | 51.5-52.1 |
| T3 | 36.0 | 93.4 | 168.6-169.4 | 12.8 | 53.4 | 66.5-67.2 |
| T2 | 36.8 | 119.0 | 225.7-226.5 | 13.6 | 61.3 | 78.5 |
| **LB** (TinyTL-L+B) | 37.1 | **244.9** | **557.2-559.8** | 13.9 | 78.7 | 122.1-123.2 |
| TinyTL-B | 31.7 | 201.8 | 537.0-537.1 | 8.5 | 43.1 | 101.7-102.4 |
| FT-Full | 390.8 | 465.5 | 598.7-599.7 | 65.5 | 152.7 | 158.6-160.0 |
| LB, GN → frozen affine | | 72.4 | 385.2 | | 57.0 | 100.6 |
| **B, GN → frozen affine** | | **29.4** | 364.9 | | 21.4 | 79.2 |
| T3, GN → frozen affine | | 54.9 | 130.0 | | 48.4 | 62.2 |

Ranges are across three processes. The first arm measured in a process pays a one-time workspace: `mem_repeat_result.json` shows LB at 573.4 when measured first and 557.2 when measured sixth in the same process.

**Lever sizes in each accounting:**

| lever | profiler | engine | PyTorch |
|---|---|---|---|
| method type, FT-Full / LB (b8) | 10.5× | 1.90× | 1.07× |
| depth, LB / T3 (b8) | 1.03× | 2.6× | 3.3× |
| depth, LB / FT-Last (b8) | 1.17× | 10.1× | 8.3× |
| batch, LB b8 / b1 | 2.7× | 3.1× | 4.6× |

**Reading.**
- The depth and batch levers are real in both non-profiler accountings. Notably, **the profiler misses the depth lever entirely**: its L+B peak is an early inference transient, 3.3 MB per image.
- Only the method-type lever depends on the accounting.
- The GN ablation pins down the profiler's error. With frozen-statistics normalization, an ideal engine needs 29.4 MiB for TinyTL-B, matching the profiler's 31.7 MB. With the GroupNorm TinyTL actually uses (adopted for batch-1 training), it needs 202 MiB. **The GN that makes small batches work is exactly what TinyTL's memory model leaves out.** In PyTorch, GN costs 172-175 MiB at batch 8 (LB 559.8 → 385.2).
- **New one-line story:** *TinyTL's 37 MB assumes normalization that is free to back-propagate through. With its own GroupNorm, the real cost is set by how deep the gradient goes and by the batch size, under an ideal engine and in PyTorch alike. We measure how deep few-shot adaptation must go in a near and a far domain, and whether batch 1 is the better way to fit a 128 MiB budget.*
- The device argument (§1) now cites the engine and measured columns and drops "exactly two levers" (S11).

### B4. Accept, and strengthened

Upstream's own batch-1 scripts show that r1's ratio rule was not a recipe:

| dataset | batch-1 LR | batch-8 LR | ratio |
|---|---|---|---|
| Flowers | 3e-4 | 2e-4 | 1.5 |
| Aircraft | 3e-4 | 16e-4 | 0.19 |
| Car | 2e-4 | 8e-4 | 0.25 |

(`upstream\tinytl\exp_scripts\batch1\*.sh` vs `batch8\*.sh`.)

r2 rules:
- LB-b1 uses the **published batch-1 LR of its dataset**: Aircraft 3e-4, or Flowers 3e-4 in the fallback.
- **Pre-registered LR check.** If H2 is *moderate*, *catastrophic* or *size unresolved*, run seeds 0-1 of LB-b1 at 1e-4 and at 1.2e-3 (÷3 and ×4; 4 runs, 2.2 h, conditional).
  - Let Δ = mean over those seeds of val(LB-b8) − val(LB-b1, published LR), at the final epoch.
  - The negative label stands only if neither alternative LR reduces Δ by more than half. Otherwise we report "the batch-1 loss is LR-sensitive" and make no degradation claim.
- **Scope.** H2 tests TinyTL's own configuration: ImageNet-pretrained GN lite residuals with L+B. TinyTrain's TinyTL baseline presumably used freshly inserted, lite-residual-only adapters on its meta-trained backbones, so H2 does not reproduce TinyTrain's condition. The closest proxy (S9) is optional O4.

### B5. Accept

The check `train ≥ 95%` together with `val gain < 1 pp over epochs 40-50` cannot fail under cosine annealing, as the reviewer showed. It is kept only as a gross-failure flag. Two changes:

1. **H3 (k=2 vs k=5) leaves the core** (O2). When it runs, it includes the reviewer's step-matched control: LB and FT-Last at k=2 for 125 epochs (3,200 steps), 3 seeds. If step-matching changes g(2) by more than 1 pp, H3 is reported under both budgets and not read as a data effect.
2. **Core convergence check that can fail.** In each LR pilot, at the chosen LR, compute Δ_half = val(epoch 50) − val(epoch 25), averaged over the pilot seeds.
   - At epoch 25 the cosine LR is still 50% of peak, so a converged arm gains little while a slow learner keeps gaining. On the old runs, the 2-seed means are 0.98 / 0.89 for r2 and 1.62 for r4 (seeds 0-1).
   - If Δ_half > 1.5 pp, run that arm on the pilot seeds at 100 epochs (≈0.5 h per arm). If val improves by ≥ 1 pp, H1 results for that arm are reported as "at the official 50-epoch budget", with the 100-epoch gain stated beside them.
   - H1 compares arms at the same 3,200 steps (Flowers) or 3,125 steps (Aircraft), so there is no step confound inside H1. H2 is epoch-matched by design, which equalizes per-image compute.

### B6. Partly accept, partly rebut

- **Accepted.**
  - The "exhaustive sweep" and "upper bound on any selection rule" wording is deleted.
  - The Fisher diagnostic is core, pre-registered as descriptive (§3, H4).
  - It uses an imprinted, ProtoNet-equivalent head: W_c = 2μ_c, b_c = −‖μ_c‖², where μ_c are class means of un-augmented penultimate features. These logits equal negative squared Euclidean distances up to a per-sample constant, which matches TinyTrain's ProtoNet setting.
  - With two datasets, task adaptivity can now be checked at least minimally: does Fisher pick a deeper cutoff on Aircraft than on Flowers?
- **Rebutted: T2 in the Flowers core.**
  - Its Flower prior is −0.6 to −1.0 pp, the least uncertain rung.
  - Adding it to the Holm family would make it a fourth test, lowering power for T3, T4 and FT-Last.
  - Its rationale was the "exhaustive" claim, which is now withdrawn.
  - T2 stays a conditional, exploratory arm on Flowers (it runs if T3 is *inferior by more than the margin*).
- **On Aircraft**, where T2 matters (prior −2.1 to −3.1), T2 is the budgeted follow-up arm. It is near-certain to run because T3 is predicted not to retain.

## Non-blocking items

| item | response |
|---|---|
| S1 | **Accepted.** `macs.py` (conv + linear, 224 px, per image). Backward = grad-input MACs for every layer behind the cutoff plus grad-weight MACs for trainable layers. FT-Last 0.1 M, T4 125 M, T3 227 M, T2 298 M, LB 459 M, B 384 M, Full 778 M; forward is 395 M for every arm. The H2b tie-break now uses memory plus backward MACs (T4 needs 3.7× fewer per image than LB-b1). GPU ms are reported only as "our machine". |
| S2 | **Accepted.** If the best LR is the edge of the grid, extend ×4 or ÷4 in that direction, at most twice per arm. |
| S3 | **Accepted.** Reseed `torch`/`numpy`/`random` right before `run_manager.train()`. Monkeypatch `tinytl.utils.common_utils.KMeans` with `random_state=0`, so the frozen layers common to two arms are quantized identically. Upstream is left unmodified. |
| S4 | **Accepted, all.** Profiler `batch_size=args.batch` and `require_backward=(arm != 'last')`; parse `Valid [e/50]` and `Train top1`; `set_trainable` before quantization and before `RunManager`. |
| S5 | **Accepted.** Instance-level `save_model` monkeypatch writes weights only at the last epoch. |
| S6 | **Done.** `mem_repeat_result.json` holds the per-position values and nvidia-smi output. Utilization is logged before each night as well. |
| S7 | **Accepted.** F.1's 30-44% describes the candidate set Fisher scores, so T3 is "the region TinyTrain's selection searches". "must" becomes "may". "label-dependent" is replaced by the specific confounds. |
| S8 | **Accepted.** r1's 17.3 h × 1.3 = 22.5 h. The paper-protocol reproduction is 3.7 h (O6). Utilization is checked before each night; another application held 11-36% of the GPU during this round's measurements. |
| S9 | Optional O4, ranked above the other extensions. |
| S10 | **Done** (Table M2). |
| S11 | Out of scope, by choice: activation recomputation, low-precision saved activations, and feature caching for FT-Last (which would make FT-Last's training almost free). |
| S12 | **Accepted.** Every label is printed with d and its 93% CI. |

---

## 1. Research question and why it matters

**Question.** At k = 5 labels per class, how shallow can TinyTL-L+B's backprop cutoff be while staying within 2 pp of full depth, in a near domain (Flowers102) and a far domain (FGVC-Aircraft)? And on the far domain, is batch 1 a better way than truncation to fit a 128 MiB budget?

**Why.**
- **The device.** TinyTrain's target is a Pi Zero 2-class device with 512 MB. At TinyTL's published batch 8, LB's training step needs **245 MiB under an ideal engine and 558 MiB in PyTorch**. The profiler's 37 MB holds only for frozen-statistics normalization (Table M2).
- **The levers.** Fitting a quarter or half of the device (128 or 256 MiB) means cutting depth or batch. Both cut memory by 2.6-10× under the engine and in PyTorch.
- **Where depth matters.** TinyTrain's AdapterDrop rungs say depth is nearly free on Flowers (all rungs within 1.9 pp) and expensive on Aircraft (17.6 pp from TinyTL down to LastLayer). That protocol differs from TinyTL's own in five ways (B1), and it reports no memory.
- **What we deliver.**
  - A paired, margin-based answer per domain, with memory in all three accountings.
  - A matched-memory test of the batch lever in the domain where truncation is expected to be costly.
  - A direct test of TinyTL's batch-1 GN claim against TinyTrain's "degrades catastrophically".

## 2. Story: kept, changed, dropped

The r1 decisions stand except where noted.

| story element | r2 decision | reason |
|---|---|---|
| ①② motivation; activations dominate | Kept | Of LB's 558 MiB, 512 are saved tensors; under the engine, 195 of 245. |
| ③ TinyTL mechanism; 95.20 vs 95.5 | Kept | One seed, test-selected; labeled a sanity reproduction. |
| ③ "37 vs 391 MB" | **Changed** | Replaced by the three accountings and the GN finding (B3). |
| ④ data scarcity, k-shot | Kept, at k=5 | The k-trend is optional O2 (B5). |
| ⑤ "TinyTL clearly beats LastLayer" | Changed | True on the nine-domain average and on far domains such as Aircraft (70.3 vs 52.7), not on Flowers (86.2 vs 85.5). Both cases are now tested. |
| ⑤ "important layers lie late" | Kept, re-sourced and narrowed | Source is F.1 plus AdapterDrop Table 6, now the H1 priors. F.1 describes the candidate set, not the trained layers (S7). |
| ⑤⑥ batch-size contradiction | Kept as H2, on Aircraft, at the published LR, with an LR check | B4. |
| ⑦ task-adaptive Fisher selection | **Changed** to a depth ladder plus a core Fisher-under-budget diagnostic on 2 tasks | Measured memory is set by the earliest selected stage, so any selection costs as much as truncating at its earliest stage. Fisher is evaluated as a cutoff chooser (H4). |
| ⑦ imprinting | Used as the Fisher head (core, no training); as a training init it is optional O9 | B6. |
| ⑧ "same budget, fixed vs adaptive, ± imprinting" | Replaced | Per-domain Pareto, Holm NI tests, H2b at 128 MiB. |
| Risk: Flowers only | **Resolved** | Aircraft is in the core. Fallback if the data cannot be obtained: Flowers-only, framed as the near-domain case (§8). |
| r1 extras (k=2, k=10, LB-b1 on Flowers) | Optional (O2, O7, O3) | Budget. |

## 3. Hypotheses and decision rules

**Common definitions.**
- Metric: final-epoch (epoch 50) top-1 on the test split (Flowers 6,149 images, Aircraft 3,333).
- d_X = acc(X) − acc(LB-b8), paired by seed (same images, same init stream).
- Labels: the four of B2, from two-sided 93% t-CIs, with margin δ = 2 pp. δ is the smallest gap the TinyTL paper treats as decisive (B 93.5 vs L+B 95.5).
- Every label is reported with d and its CI.

### H1-F (Flowers, n = 10 → 15)

Holm over {T3, T4, FT-Last}.

| prior d (pp) | prediction (MC at paired SD 1.8) |
|---|---|
| T3 −1.0 | retains 54% (32% after Holm) |
| T4 −1.9 | size unresolved 91% |
| FT-Last −0.7 | retains 74% (47% after Holm) |

P(some arm triggers the extension) is 0.44 at SD 1.8 and 0.80 at SD 2.5. On average 0.55 arms (SD 1.8) to 1.28 arms (SD 2.5) are extended.

### H1-A (Aircraft, n = 5)

Holm over {T3, FT-Last, plus T4}. T4 is core on Aircraft because H2b needs it. The extension is optional O1.

| prior d (pp) | prediction (MC at paired SD 2.5) |
|---|---|
| T3 −3.1 | size unresolved 79%, inferior 17% |
| T4 −10.6 | inferior about 100% |
| FT-Last −17.6 | inferior about 100% |

**Aircraft follow-up arm (core).** If T3 does not *retain*, run T2 (exploratory) to locate the needed depth. The prior makes this near-certain. If T3 retains, no follow-up is needed (T4 is already in the core) and the 0.87 h is released.

### Cross-domain statement (descriptive)

d_FT-Last(Aircraft) − d_FT-Last(Flowers), and the same for T3 and T4, each with a Welch CI. The priors predict a gap of about 17 pp for FT-Last, about 9 for T4 and about 2 for T3. The first two resolve trivially; T3 may not.

### H2 (Aircraft): GN TinyTL at batch 1

- Comparison: d_b1 = acc(LB-b1, published LR 3e-4) − acc(LB-b8), seeds 0-4 (n = 5). If the result is *inconclusive*, extend to n = 8.
- Labels:

| label | rule |
|---|---|
| *no meaningful degradation* | L > −2 |
| *catastrophic* | U < −5 |
| *moderate* | −5 ≤ U < −2 |
| *size unresolved* | U < 0, L ≤ −2, U ≥ −2 |
| *inconclusive* | L ≤ −2, U ≥ 0 |

- Negative labels require the B4 LR check before they are reported.

MC at paired SD 2.5 (`power_r2b.py`):

| true d | P(extension fires) | final labels |
|---|---|---|
| 0 | 0.62 | no degradation 0.61, inconclusive 0.33 |
| −10 | | catastrophic 0.95 |
| −6 | | moderate 0.70 |

So "catastrophic or not" is well resolved; non-inferiority within 2 pp is weak at this n, and the report will say so.

### H2b (Aircraft, derived): which lever fits 128 MiB better

- Comparison: e = acc(LB-b1, 122.6 MiB) − acc(T4-b8, 105.1 MiB), seeds 0-4, 93% CI.
  - *Batch lever better* if L > 0.
  - *Depth lever better* if U < 0.
  - Otherwise a tie, which goes to T4: 14% less measured memory and 3.7× fewer backward MACs per image.
- MC: if LB-b1 retains and T4 is at its prior (e ≈ +10.6), P(batch better) is 1.00 at SD 2.5 and 3.5. If e = +5 it is 0.95 (SD 2.5).
- At 256 MiB, LB-b1 vs T3-b8 (169 MiB) and T2 (226 MiB) are read off the Pareto plot with the same CI.

### H4 (both datasets, descriptive, no training): Fisher-under-budget as a cutoff chooser

Per seed, compute TinyTrain's Fisher potential (Eq. 2) on the k-shot images through the imprinted ProtoNet head, before any training. Stages are grouped to match the cutoffs we run: G2 = stage 2, G3 = stage 3, G45 = stages 4-5. Each group is scored with Eq. 3, normalized by the group's trainable parameters and the backward MACs it adds (`macs.py`).

TinyTrain's prefix-greedy rule then runs under 128 and 256 MiB budgets, using Table M2's measured batch-8 memory of the truncation at the earliest selected group. Its result is mapped to a cutoff c_F.

Reported:
- the hit rate and mean regret of c_F against the per-seed oracle, i.e. the best feasible run arm by test accuracy;
- the same for the static rules "deepest feasible" and "always FT-Last";
- the fraction of seeds where c_F at 256 MiB is deeper on Aircraft than on Flowers (task adaptivity).

Pre-registered reading:
- **"Useful"**: Fisher's mean regret is ≤ that of the better static rule on both datasets, and it picks deeper on Aircraft in ≥ 70% of seed pairs.
- **"No advantage"**: otherwise.

This is descriptive: 5-10 seeds and 3-4 options per budget.

## 4. Conditions

Every arm runs the same network with all lite residuals in the forward pass; arms differ only in what receives gradients (Table M2).

| block | dataset | arm(s) | batch | LR | seeds | runs | status |
|---|---|---|---|---|---|---|---|
| P-F | Flowers | LB, T3, FT-Last | 8 | {2e-4, 8e-4} + edge ×4 / ÷4 (at most 2 extensions) | 100, 101 | 12-24 | core |
| A-F | Flowers | LB, T3, T4, FT-Last | 8 | per pilot (T4 inherits T3's LR if T3 and FT-Last agree, else 4 pilot runs) | 0-9 | 40 | core |
| P-Air | Aircraft | LB, T3, FT-Last | 8 | {4e-4, 1.6e-3, 6.4e-3} + edge | 100 | 9-15 | core |
| A-Air | Aircraft | LB, T3, T4, FT-Last | 8 | per pilot | 0-4 | 20 | core |
| C-Air | Aircraft | LB-b1 | 1 | 3e-4 (published) | 0-4 | 5 | core |
| Air-follow | Aircraft | T2, only if T3 does not retain | 8 | T2 inherits LB's LR if LB and T3 agree, else T3's | 0-4 | 5 | core (near-certain) |
| F-ext | Flowers | LB + inconclusive arms | 8 | as A-F | 10-14 | 10-20 | likely conditional (P = 0.44-0.80) |
| C-ext | Aircraft | LB-b1, plus LB-b8 at the same seeds for pairing | 1 / 8 | 3e-4 / as A-Air | 5-7 | 3 + 3 | likely conditional |
| H2-LR | Aircraft | LB-b1 at 1e-4 and 1.2e-3 | 1 | | 0-1 | 4 | conditional (negative H2 only) |
| half-budget | either | flagged arm at 100 epochs | 8 | chosen | pilot seeds | 2 per arm | conditional |
| F-T2 / F-T5 | Flowers | T2 (if T3 inferior) / T5 (if FT-Last does not retain and T4 retains) | 8 | inherit | 0-9 | 10 | conditional, exploratory |

On Aircraft, T4 inherits the same way as on Flowers, but with 1 seed (2 runs). The 3 LB-b8 runs at seeds 5-7 are needed only if C-ext fires, and are counted in C-ext's cost.

## 5. Protocol

**Flowers.** Unchanged from r1:
- original `setid.mat` splits;
- k = 5 per class from the train split with `Random(seed)`;
- the val split (1,020) is used only for the pilot and diagnostics;
- test (6,149) is evaluated once.

**Aircraft data prep** (new; CPU only, Day 0):

| step | details |
|---|---|
| Download | FGVC-Aircraft 2013b, `fgvc-aircraft-2013b.tar.gz` from the Oxford VGG site, **about 2.7 GB**. `torchvision.datasets.FGVCAircraft(download=True)` fetches the same archive. The team downloads it; this needs their approval. Disk has 83 GB free. |
| Splits (the setid equivalent) | `data/images_variant_{train,val,test}.txt`: 3,334 / 3,333 / 3,333 images, 100 variants, about 33 per variant per split. This is the variant level, 100 classes, matching upstream `n_classes = 100`. |
| `prepare_aircraft.py` (≈ 70 lines, ≈ 10 min CPU) | Crop the 20-pixel copyright banner at the bottom (standard, declared). Resize so the shorter side is 512 px, JPEG quality 95. Every RandomResizedCrop(224, scale ≥ 0.22) is still a downsample (smallest crop side ≈ 290 px), and test Resize(256)/CenterCrop(224) is unchanged. Write `data/aircraft_prep/{train,val,test}/<variant>/<id>.jpg`. |
| Why pre-resize | Measured CPU cost per training image (`decode_time.py`): Flowers 5.2 ms; Aircraft-sized 1.3 MP JPEG 13.6 ms, or 17.3 ms with upstream's ColorJitter; pre-resized 754×512 with jitter 10.3 ms. Pre-resizing keeps Aircraft runs within about 20% of Flowers run time. |
| k-shot mapping | `build_root_aircraft(k, seed)` mirrors `build_root`: k per variant from **train** with `Random(seed)`. **val** is a fixed 10-per-variant subset of the val split (1,000 images, chosen with `Random(12345)`), the same size as Flowers val. **test** is the full 3,333. |
| Provider | Monkeypatch `AircraftDataProvider.save_path` to read `TINYTL_AIRCRAFT_PATH`, like Flowers. Upstream is not modified. |
| Recipe | Upstream `exp_scripts/batch8/aircraft.sh`: 50 epochs, Adam, cosine, label smoothing 0.7, `distort_color torch`, batch 8. The LR comes from the pilot, centred on upstream's 16e-4. Batch 1 uses `batch1/aircraft.sh`'s 3e-4. |

**Model selection.** None; the epoch-50 model is evaluated on test once. Val is logged every 5 epochs for the half-budget check and learning curves.

The protocol bridge (old r2 final checkpoints on test) stays, documentation only.

**Learning rate.**
- Per arm and dataset: highest mean final-epoch val on the pilot seeds.
- A higher LR replaces a lower one only if it wins by ≥ 1 pp; otherwise the LR closest to upstream's recipe is kept.
- Edge rule (S2): extend in the winning direction, at most twice.
- Aircraft uses 1 pilot seed (budget). Within a seed the LR comparison is paired, and a 4× LR step usually moves val by several pp when LR matters.
- `lr_choice.json` is frozen before the A blocks.

**Training budget.** The official recipe, 50 epochs: 3,200 steps on Flowers and 3,125 on Aircraft at batch 8; 25,500 / 25,000 at batch 1. Train ≥ 95% is a gross-failure flag; the half-budget check is the convergence check (B5).

**RNG** (S3).
- Reseed all generators immediately before `train()`.
- KMeans `random_state=0`.
- A seed therefore fixes the subset, the head init and the data order across arms by construction.

**Memory.** `measure_memory.py`, i.e. `mem_harness.py` + `engine_bound.py` moved into the project and sharing `set_trainable` with the trainer.
- Each process starts with a throw-away arm.
- Three processes, run in different arm orders.
- Table M2 with all three accountings and the GN ablation.
- Never taken from training runs; the run peak is stored as `run_peak_incl_eval_MiB`.
- Aircraft uses the same network and 224 px input, so Table M2 applies (the 100- vs 102-way head is a negligible difference).

**Compute.** Analytic MACs from `macs.py`. GPU ms per step only on an idle GPU (`measure_time.py` refuses to run above 5% utilization), labeled "our machine".

## 6. Run list, GPU-hours, order and gates

**Per-run cost.** The run-time model (16.8 ms per Flowers training image for LB, including loading; +1.8 min per run for val, test and setup) uses arm-specific GPU time. Measured: LB 7.2, T3 5.0 and head 3.1 ms per image at batch 8. Interpolated: T4 about 4.1, T2 about 6.1. Aircraft adds 5.1 ms per image of decode and jitter (measured) plus about 0.2 min of eval.

| run type | minutes per run |
|---|---|
| Flowers k=5 b8: LB / T2 / T3 / T4 / FT-Last | 9.0 / 8.5 / 8.0 / 7.7 / 7.2 |
| Aircraft k=5 b8: LB / T2 / T3 / T4 / FT-Last | 10.9 / 10.4 / 10.0 / 9.6 / 9.2 |
| LB-b1 k=5: Flowers / Aircraft | 30.7 / 32.5 |

The conservative column applies LB's rate to every arm.

| order | block | runs | realistic GPU-h | conservative GPU-h | gate after it |
|---|---|---|---|---|---|
| Day 0 | Aircraft download and prep (CPU); code | n/a | 0 | 0 | **G-data:** Aircraft roots built and one smoke epoch passes, otherwise switch to the fallback (§8). |
| 1 | M: memory ×3 processes, engine bound, MACs, timing (idle GPU), bridge | n/a | 0.15 | 0.15 | G0: trainable counts equal Table M2; smoke run of every (dataset, arm, batch). |
| 2 | P-F (12 base + about 4 edge) | 16 | 2.15 | 2.4 | Half-budget check. |
| 3 | P-Air (9 base + about 1 edge) | 10 | 1.67 | 2.2 | Freeze `lr_choice.json`. **G-time:** first-run setup and minutes per run vs the model. |
| 4 | A-F: seed-major | 40 | 5.32 | 6.0 | |
| 5 | A-Air: seed-major | 20 | 3.31 | 3.63 | **G1:** Holm labels; decide the Air follow-up arm and the F-ext arms. |
| 6 | C-Air | 5 | 2.71 | 2.71 | **G2:** H2 label; C-ext, or the H2 LR check. |
| 7 | Air follow-up (T2 if T3 does not retain) | 0-5 | 0.87 | 0.91 | |
| 8 | Fisher diagnostic (H4) | n/a | 0.1 | 0.1 | |
| | **core** | 96-106 | **16.3** | **18.1** | |
| 9 | F-ext (LB + 1 arm × 5) | 10 | 1.4 | 1.5 | likely |
| 10 | C-ext (+3 seeds of LB-b1 at 32.5 min, +3 seeds of LB-b8 at 10.9 min for pairing) | 6 | 2.2 | 2.2 | likely |
| | **core + likely conditionals** | | **19.8** | **21.8** | |
| | Unlikely conditionals: F-ext extra arms (≤ +1.3), H2 LR check (2.2), half-budget checks (≤ 0.5 each), F-T2 or F-T5 (1.4 each), extra pilot edges (≤ 1.7) | | ≤ 8.5 | | |

**Pre-registered cut order.** Applied if G-time shows more than +10% vs the model, or if the core plus likely conditionals would pass 20 h:
1. C-ext becomes 2 seeds (1.45 h);
2. F-ext covers only the arm with the smallest Holm p among the inconclusive ones;
3. the Air follow-up arm moves to optional;
4. A-F drops to 8 seeds.

**Schedule.**
- Day 0: prep.
- Day 1: M, P-F and P-Air (about 4 h).
- Night 1: A-F and A-Air (8.6 h).
- Day 2: G1 analysis and Fisher.
- Night 2: C-Air, the follow-up arm and F-ext (5.0 h).
- Day 3: G2, then C-ext (2.2 h).

**Optional, ranked.** Run in order while the cumulative total stays ≤ 40 h.

| rank | block | GPU-h |
|---|---|---|
| O1 | Aircraft extension, +5 seeds for LB and the inconclusive arms | 1.75 |
| O2 | H3 k-trend on Flowers: k=2 × {LB, T3, FT-Last} × 10, plus the step-matched control | 2.95 |
| O3 | H2 on Flowers: LB-b1 × 6 | 3.1 |
| O4 | S9: TinyTL-L with randomly initialized lite residuals, batch 1 vs 8, 4 seeds, Aircraft | 2.8 |
| O5 | FT-Full at k=5 on both datasets (2-LR pilot, plus 10 / 5 seeds) | 3.9 |
| O6 | Paper-protocol reproduction of the 5 baselines | 3.7 |
| O7 | k=10 on Flowers, descriptive | 2.4 |
| O8 | Combined levers T3-b1 / T4-b1 on Aircraft × 5 | 4.7 |
| O9 | Imprinting as a training init at k=2 | 1.6 |

## 7. Implementation plan

About 2-2.5 person-days. Nothing existing is modified; upstream behaviour is changed only through monkeypatches inside our scripts.

| # | file | what | lines |
|---|---|---|---|
| 1 | `fewshot_depth.py` (from `fewshot_resolution.py`) | Everything listed in r1 §7 #1, plus: `--dataset {flowers102, aircraft}` with per-dataset roots and recipes (`distort_color`); `AircraftDataProvider.save_path` patch; reseed before `train()`; KMeans `random_state` patch; profiler `batch_size=args.batch` and `require_backward`; fixed parsing of the val/train log; final-epoch-only `save_model` patch; result fields `dataset` and `run_peak_incl_eval_MiB`. | ≈ 270 (≈ 110 new or changed) |
| 2 | `prepare_aircraft.py` + `build_root_aircraft` | §5 data prep and k-shot roots. | ≈ 90 |
| 3 | `measure_memory.py` | Merges `mem_harness.py`, `engine_bound.py` and `mem_repeat.py`: warm-up arm, 3 processes, Table M2 with the GN ablation, JSON output with utilization. | ≈ 200 |
| 4 | `measure_macs.py`, `measure_time.py` | From `macs.py`; timing refuses to run on a busy GPU. | ≈ 40 + 45 |
| 5 | `make_runlist.py`, `run-depth-sweep.sh` | As in r1, with dataset and block columns, gates as stop points, and the cut order encoded as flags. | ≈ 90 |
| 6 | `fisher_stage.py` | ProtoNet-equivalent head from un-augmented features; hooks on block outputs; Δo = (1/2N) Σ_n (Σ_d a·g)² per channel; group sums; Eq. 3 normalization; prefix-greedy under the measured budget; regret vs oracle and the static rules. | ≈ 110 |
| 7 | `analyze_depth.py` | 93% CIs, the four labels, Holm, the extension bookkeeping, Welch cross-domain contrasts, H2/H2b labels with the LR-check rule, Pareto plots per dataset with all three memory columns, prior-vs-observed table. | ≈ 220 |
| 8 | `eval_final_ckpt.py` | Bridge, as in r1. | ≈ 40 |

Suggested split:
- Person 1: #1 and #8.
- Person 2: #2, #5 and the Aircraft download.
- Person 3: #3, #4, #6 and #7.

## 8. Risks, fallbacks and expected outcomes

| risk | fallback |
|---|---|
| Aircraft download or prep fails by the end of Day 0, or the archive is unavailable | Pre-registered **fallback (b)**: Flowers only, framed as the near-domain case where the prior predicts small depth effects. C moves to Flowers (LB-b1 × 6 at 3e-4, 3.1 h); O2 (k=2 + step-matched, 2.95 h) moves into the core; the Aircraft budget (≈ 8.6 h) covers these. |
| Aircraft paired SD > 3 | T4 and FT-Last still resolve (priors ≫ margin). T3 stays *size unresolved* or *inconclusive* (O1 extends it). H2's "catastrophic or not" stays resolved (P ≥ 0.75 at SD 3.5, d = −10). |
| Aircraft LR pilot with 1 seed picks badly | The edge rule plus the 1 pp tie rule. If an H1-A label hinges on an arm whose pilot margin was < 1.5 pp, spend 2 sensitivity runs from the unlikely-conditional budget. |
| Run-time model off | The G-time gate plus the cut order. |
| Batch-1 LR confound | The B4 check; no negative H2 claim without it. |
| GPU shared with other apps | Check utilization before each night (11-36% was seen this round). Memory is unaffected; time is. |
| Priors do not transfer (different protocol) | That is a finding. Report the prior-vs-observed table, and attribute any gap only to the five listed protocol differences, with "may". |

**Expected outcome → conclusion**

| result | conclusion |
|---|---|
| H1-F: FT-Last *retains* (Holm) | On near-domain few-shot data, truncating all the way to the classifier costs < 2 pp: 8.3× less measured memory (10.1× under the engine). TinyTL's adaptation is not worth its memory here. Consistent with Table 6. |
| H1-F: FT-Last *size unresolved*, T3 retains | The classifier alone costs a real but bounded amount (≤ \|L\| pp). Stages 3-5 recover it at 169 MiB (3.3×). |
| H1-F: any arm *inferior by more than the margin* | Depth matters even on Flowers under TinyTL's recipe, against Table 6. Run F-T2 (if T3 is the inferior arm) to locate the needed depth. |
| H1-A: T4 and FT-Last *inferior*, T3 size unresolved or inferior, T2 retains | On far-domain data the last 3 stages are not enough; stages 2-5 are (226 MiB, 2.5×). Matches Table 6's ladder. |
| H1-A: T2 also not retained | Far-domain few-shot needs (nearly) full-depth adaptation: truncation buys at most about 1.7× before costing > 2 pp, so only the batch lever (H2) remains. |
| H1-A: T3 *retains* | TinyTL's pretrained lite residuals and L+B make shallow adaptation work where Table 6's fresh adapters did not. That is a protocol effect worth reporting. |
| H2: *no meaningful degradation* | With its GN checkpoint, TinyTL trains at batch 1 in few-shot within 2 pp at 123 MiB. TinyTrain's claim does not hold for TinyTL's own configuration; its cause may lie in their setup (O4 probes it). |
| H2: *catastrophic*, confirmed by the LR check | Even with GN, batch 1 fails few-shot TinyTL; TinyTrain is reproduced. On far-domain data a ≤128 MiB device then forces a loss of > 2 pp. |
| H2: negative but the LR check recovers > half | The batch-1 loss is an LR artifact; no degradation claim. Report the recovered LR as the few-shot batch-1 recipe. |
| H2b: batch better | On far-domain data, fit 128 MiB with batch 1, not truncation (expected margin about 10 pp if the priors hold). |
| H2b: depth better or tie | Fit 128 MiB with T4. |
| H4 "useful" / "no advantage" | Fisher-under-budget picks a deeper cutoff where depth matters at no regret, so the task-adaptive idea of the story survives. Otherwise, a static rule ("deepest that fits") is as good, so the story's idea is not supported at this granularity. |
