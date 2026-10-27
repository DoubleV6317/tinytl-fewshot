# Review of proposal_r1.md (round 1)

**Verdict: MAJOR REVISION.**

Block A (k=5, paired, seed-major, final-epoch, per-arm LR) is sound and should stay. But two headline claims are wrong as written: "nobody has measured it inside TinyTL", and "TinyTL's memory saving is a property of its analytical model". Two of the decision-rule labels say something the rule does not test. Two of the three secondary hypotheses (H2 and H3) are unprotected against the confound most likely to produce a negative result. Most fixes are cheap, but together they change the run list and the budget, so a second round is needed.

What I checked:
- I re-ran `power_calc.py`.
- I wrote `design\review_mc.py` and `design\review_mc2.py`: Monte Carlo of the H1/H2 rules, including variability of the sample SD and the extension look.
- I verified the budget against the run-time model and against the timestamps of the old r2 runs (`fewshot-sweep.log`).
- I read `fewshot_resolution.py`, `tinytl_fgvc_train.py`, `run.py`, `modules.py`, `common_utils.py`, `memory_cost_profiler.py`, the OFA `run_manager.py`, `run_config.py` and `imagenet.py`, and `my_modules.get_parameters`.

---

## Blocking issues

### B1. The closest prior evidence is missing: TinyTrain Table 6 already measured truncated TinyTL ("AdapterDrop"). This breaks the novelty claim and makes the dataset choice the weakest case.

**Evidence**
- `tinytrain_paper.txt` lines 1060-1065 describe AdapterDrop. It drops the adapters of the first, shallow blocks, so that "backpropagation is not required all the way to the input layer". The paper calls TinyTL "one variant of AdapterDrop with no dropped layers".
- Table 6 (lines 1233-1239) gives ProxylessNAS results. AdapterDrop-75/50/25% dropped is roughly T4 / between T2 and T3 / T1.

| method | Flower | Aircraft | 9-domain average |
|---|---|---|---|
| LastLayer | 85.5 | 52.7 | 57.0 |
| AdapterDrop-75% | 84.3 | 59.7 | 59.1 |
| AdapterDrop-50% | 85.2 | 67.2 | 61.9 |
| AdapterDrop-25% | 85.6 | 68.2 | 62.8 |
| TinyTL | 86.2 | 70.3 | 63.7 |
| FullTrain | 86.3 | 71.4 | 65.0 |

**Consequences**
1. Proposal line 61, "nobody has measured it inside TinyTL", is false as written. The real gaps are different:
   - TinyTrain reports no memory for AdapterDrop.
   - It uses meta-trained backbones and freshly inserted lite residuals with no bias updates, not TinyTL's ImageNet-pretrained lite-residual checkpoint with L+B.
   - It uses various-way tasks.
   - It gives no paired seed-level inference against a margin.
2. On Flower, every depth in Table 6 lies within 1.9 pp of TinyTL (84.3-86.3), and LastLayer is 0.7 pp below TinyTL. The prior therefore predicts that H1 at δ = 2 ends in "everything retains", the least informative outcome. On Aircraft the same ladder spans 17.6 pp and is monotone. This is exactly the dataset where the depth question has teeth.

The story itself listed this risk (story.txt, risks section). The proposal acknowledges it in §2 but ranks Aircraft last (O6, line 269).

**Required**
- Cite AdapterDrop and Table 6 in §1/§2, and restate the contribution against it:
  - measured (not modelled) memory of truncation;
  - TinyTL's own checkpoint and protocol;
  - paired inference with a margin;
  - the batch lever.
- Write the Table 6 Flower/Aircraft numbers in as the pre-registered prior predictions for H1.
- Then either:
  - **(a)** promote O6 into the core: Aircraft k=5, {LB, T3, FT-Last}, 5 seeds, about 3.3 h. Note that upstream's Aircraft LR is 16e-4, eight times Flowers, so the pilot must be centred there (e.g. {4e-4, 1.6e-3, 6.4e-3}), not on the Flowers grid. Pay for it by moving B10 (k=10, n=3, descriptive only, 2.4 h) to optional; or
  - **(b)** keep Flowers-only, but say explicitly that it is the near-domain case where the prior predicts small depth effects. Then the main deliverable is the measured-memory Pareto plus the batch lever, not "how deep adaptation must go" in general.

Option (a) is strongly preferred.

### B2. The decision-rule labels say more than the rules test, and the stated error rates are optimistic.

**Evidence: labels**
- "Loses more than the margin" is defined as U < 0 and L ≤ −2 (lines 98-100). That is "significant loss, non-inferiority not shown", not "loss > 2 pp", which needs U < −2.
- Monte Carlo (`review_mc.py`), SD 1.8, n=10, true d = −1.0 pp:
  - the proposal's label fires **37%** of the time;
  - U < −2 fires 0.0% of the time.
- The H2 "moderate degradation" reading (line 334: "Batch 1 costs 2-5 pp") has the same flaw.
- §8 then turns these labels into recommendations, for example "T3 loses → few-shot TinyTL needs stages 2-5".

**Evidence: "inconclusive zone" column and line 111**
- The "inconclusive zone of true d" column (line 105) is actually the zone of the observed mean at s = planning SD. Line 111 says "essentially every outcome is labeled".
- With the sample SD allowed to vary, P(inconclusive at n=10) is:
  - 3.6-13.6% at SD 1.8 (for true d from 0 to −1.5);
  - 20-38% at SD 2.5;
  - 33-50% at SD 3.0.
- The planning SD rests on three paired differences (2 df). Its 90% CI is [1.06, 8.1] pp, so SD 2.5-3 is plausible, not pessimistic.
- Unpaired SDs per config are only about 1 pp (r2 1.10, r1 0.82). Pairing has not yet shown any benefit.

**Evidence: FWER claim**
- The n=10 → 15 extension is a second look at α = 0.05. Simulated type-I error of the "retains" claim at true d = −2: 5.5% (SD 1.8), 6.7% (SD 2.5), 7.2% (SD 3.0). So "controls family-wise error at 5%" (line 117) is not true.
- The conditional T2 test after T3 fails (line 122) sits outside the fixed sequence (T2 has more memory than T3). The same holds for T5 after FT-Last fails.

**Required**
- Use four labels:
  - *retains*: L > −2;
  - *inferior by more than the margin*: U < −2;
  - *significant loss, size unresolved*: U < 0, L ≤ −2, U ≥ −2;
  - *inconclusive*: L ≤ −2, U ≥ 0.

  Rewrite the §8 outcome table so that only "inferior by more than the margin" triggers "deeper adaptation is needed".
- Do the same for H2: "moderate" must mean U < −2 and U ≥ −5.
- Use a two-sided 93% CI (one-sided 3.5%) at both looks. `review_mc2.py` shows this gives type-I error of 4.2 / 5.1 / 5.2% at SD 1.8 / 2.5 / 3.0, while P(retain | d = 0) barely moves (0.975 / 0.872).
- Replace the zone column with the simulated P(inconclusive).
- Budget c3 for the possibility that every arm needs the extension.

### B3. The memory narrative over-corrects: one framework artifact (PyTorch) is swapped for another (the profiler).

**Evidence** (`mem_harness_result.json`, saved MiB at batch 8)

| arm | GN inputs | activations (fp32) | conv-saved |
|---|---|---|---|
| LB | 172.3 | 146.3 | 192.6 |
| B | 172.3 | 146.3 | 185.7 |
| Full | 172.3 | 146.3 | 207.9 |

The proposal itself says the following are framework artifacts:
- conv inputs saved for frozen weights (≈186 MiB);
- fp32 ReLU6 inputs where a 1-bit mask suffices (146 → ≈4.6 MiB).

An engine that avoids only those two artifacts would need roughly the following (my estimate; it ignores transients):

| arm | components | total |
|---|---|---|
| LB | 172 + 5 + 7 (lite-residual conv inputs) + 18 (weights) + 22 (grads + Adam) | ≈ 224 MiB |
| B | | ≈ 197 MiB |
| Full | 172 + 5 + ≈200 + 18 + 54 | ≈ 450 MiB |

That is about **2×** for Full vs LB: not 10.5× (profiler) and not 1.07× (PyTorch).

So the following statements are PyTorch facts, not facts about TinyTL:
- "which kind of parameter is trained moves almost nothing" (line 41);
- "TinyTL's memory saving is a property of its analytical model" (line 84).

The genuine profiler error is narrower and more interesting. `count_bn` (`memory_cost_profiler.py` lines 57-64) charges a norm layer only when its weight is trainable. A frozen-γ GroupNorm still needs its normalized input to pass gradients upstream. Only frozen-statistics (eval-mode) BN would be free. In other words, the GN that TinyTL adopts for batch-1 training is exactly what its memory model leaves out.

**Required**
- Report three columns per arm:
  1. profiler;
  2. measured PyTorch;
  3. an "engine lower bound" derived from the saved-tensor breakdown: GN inputs + 1-bit activation masks + conv inputs only where the conv weight is trainable + weights + optimizer state.

  This costs no training.
- Reword the one-line story and the §1 device argument. The depth and batch levers hold in all three accountings (for example LB/T3 is about 3× under both the PyTorch and the engine estimate). The method-type lever (Full vs L+B) is the one whose size depends on the accounting: 10.5× / ≈2× / 1.07×. The story should say exactly that.

### B4. H2's negative outcomes are confounded by an untested batch-1 learning rate.

**Evidence**
- LB-b1 gets 1.5 × LR_LB with no batch-1 check (line 175).
- If the pilot moves LR_LB to 8e-4 or 3.2e-3, LB-b1 runs 25,500 Adam steps at 1.2e-3 or 4.8e-3, a value nobody has run.
- §8 (line 333) turns a catastrophic result into "TinyTrain's observation is reproduced … the batch lever is unusable". An untuned small-batch LR is one of the most plausible explanations for TinyTrain's own "catastrophic" remark, so the design can reproduce the artifact it is meant to examine.
- A positive H2 result is robust to LR mis-specification; a negative one is not.

**Required.** Pre-register that a *moderate* or *catastrophic* label is reported only after a batch-1 LR check:
- seeds 0-1 at one alternative LR: the published 3e-4 if LR_LB-b1 ≠ 3e-4, otherwise 1e-4;
- the label stands only if the alternative does not remove more than half of the val-measured drop;
- cost: ≤ 4 runs, about 2 h, from the conditional budget.

Also state that H2 tests TinyTL's own configuration (ImageNet-pretrained GN lite residuals, L+B). TinyTrain's TinyTL baseline must have used freshly inserted lite residuals on its own meta-trained backbones (lite residuals only, no bias), so H2 cannot reproduce TinyTrain's condition (see suggestion S9).

### B5. The under-training check cannot fail, so H3 (k=2 vs k=5) is unguarded against the 1,300- vs 3,200-step confound.

**Evidence**
- The check is: train top-1 ≥ 95% and val rising < 1 pp from epoch 40 to 50 (line 208).
- Under cosine annealing the LR at epoch 40 is already 9.5% of peak, and 2.4% at epoch 45, so the last 10 epochs are flat by construction.
- Train top-1 says nothing about val convergence. In the old r2 runs, train top-1 is 98.6-99.6% at epoch 10 while val is still 3-5 pp below its final value (seed 0: 83.3 → 87.3).
- Even the clearly worse r4 config passes both criteria (seed 0: train 99.8%, val 79.9 → 79.5 from epoch 40 to 50).

H3 compares the LB − FT-Last gap at k=2 (1,300 steps) with k=5 (3,200 steps) using k=5-tuned LRs. Any difference in convergence speed between a 2M-parameter adapter and a linear head shows up as a "k effect".

**Required.** Replace the check with a step-matched sensitivity at k=2:
- LB and FT-Last at 125 epochs (3,200 steps, cosine over the full run) on 3 seeds;
- ≈ 9 min each, about 0.9 h total;
- pre-registered rule: if step-matching changes g(2) by more than 1 pp, report H3 under both budgets and do not interpret the k-trend as a data effect.

Keep the "train ≥ 95%" condition only as a gross-failure flag.

### B6. The reason given for dropping Fisher selection relies on an "exhaustive sweep" that the run list does not contain. Fisher selection itself is left optional.

**Evidence**
- §2 (line 76) justifies dropping the story's task-adaptive selection like this: "the cutoff is one-dimensional with 6 levels, so sweeping it … gives an upper bound on what any selection rule can achieve".
- The core actually runs T3, T4 and FT-Last; T2 and T5 run only conditionally (lines 177-179), and T0/T1 never run.
- O3, the only remaining piece of the story's "borrow TinyTrain's idea cheaply" thread (story ⑥ and ⑦), costs 0.1 GPU-h but is optional, rank 3 (line 266).
- Without it, a grader sees a depth sweep plus a batch ablation.

**Required**
- Make T2 core at k=5. That is 10 runs, about 1.5 h, and it absorbs c1. Use the fixed sequence T2 → T3 → T4 → FT-Last. Keep T5 conditional and label it exploratory, outside the FWER claim.
- Otherwise, delete the "exhaustive" and "upper bound" wording.
- Promote O3 to core with a pre-registered evaluation: per seed, does the Fisher-under-budget rule pick the cutoff (at 128 and at 256 MiB *measured*) that the sweep shows to be best, and does the per-stage Fisher ranking correlate with the per-cutoff accuracy?
- Compute the Fisher potential with an imprinted (class-mean) head, not the random head. TinyTrain computes it on a ProtoNet backbone, whose classifier is effectively prototypes, and a random 102-way head produces meaningless activation gradients. This also gives imprinting (story ⑦) a principled, nearly free role. The code is #8 (about 25 lines) and is shared with O5.

**Budget after B1-B6 (my arithmetic, same run-time model)**

| block | change | GPU-h |
|---|---|---|
| core in proposal | | 15.8-17.3 |
| B10 | moved to optional | −2.4 |
| Aircraft core | added (B1a) | +3.3 |
| T2 | added | +1.5 |
| step-matched k=2 | added | +0.9 |
| Fisher | added | +0.1 |
| **new core** | | **≈ 19.2-20.7** |

Before the +30% margin this is at the 20 h line. Conditionals (A-ext, C-ext, the H2 LR check, T5) stay within about 7 h.

---

## Non-blocking suggestions

**S1. Compute claims.** Step time on the 3060 is launch-bound (batch 1 costs the same per step as batch 8), so GPU ms ratios are not device compute. `measure_time.py` will also refuse to run while the GPU is shared (proposal §0 saw 40% utilization).
- Report analytic backward MACs per arm: grad-input MACs for every layer behind the cutoff, plus grad-weight MACs for trainable convs. TinyTrain Table 2 reports compute in MACs, so the numbers become directly comparable.
- The H2b tie-break (line 144) should rest on memory and backward MACs, not "8× GPU time per image". On a CPU or MCU device, batch 1 costs about the same per image as batch 8, and epoch matching already equalizes per-image compute (which is a strength of the design).

**S2. Edge-of-grid rule for the LR pilot.** If the best LR is the largest tried, extend by ×4 once. This matters most for FT-Last: a linear head under Adam often prefers ≥ 3e-3, and an under-tuned FT-Last would falsely end the H1 sequence with "depth needed".

**S3. RNG pairing.**
- `weight_quantization` uses sklearn KMeans with `random_state=None` (`common_utils.py` line 46). It consumes the global numpy RNG once per frozen layer, and the set of frozen layers differs by arm (`common_utils.py` lines 73-76).
- The torch stream (data order, augmentation, and the shuffled val loader in `imagenet.py` lines 96-98) looks equal across batch-8 arms.
- To make "a seed fixes the data order" true by construction, reseed `torch`, `numpy` and `random` immediately before `run_manager.train()`.
- Optionally quantize every arm with a per-layer fixed `random_state`, so the frozen backbone is bit-identical across arms.

**S4. Trainer details** (they will not break anything, but they would silently report the wrong thing):
- Pass `batch_size=args.batch` to `profile_memory_cost`. `fewshot_resolution.py` line 172 hardcodes 8, so batch-1 runs would report the batch-8 profiler number.
- Pass `require_backward=(arm != 'last')`, as upstream `tinytl_fgvc_train.py` lines 194-199 does.
- With `validation_frequency=5`, the old `best_epoch = index + 1` (line 195) is wrong. Parse `Valid [e/50]` for the epoch and `Train top1 ([\d.]+)` for train accuracy. The existing regex `top1 ([\d.]+)` matches the val number first.
- `set_trainable` must stay before `weight_quantization` and before `RunManager(...)`. `MyNetwork.get_parameters` filters on `requires_grad` when the optimizer is built (`my_modules.py` lines 214-235), and quantization picks frozen layers by `requires_grad`. The plan as written, replacing lines 164-170, satisfies this; keep it that way.
- Deleting lines 196-197 does give the in-memory epoch-50 model (`run_manager.py` lines 342-367). The bridge can rebuild the r2 network with the official insertion: I checked that the r=2 custom insertion is identical to `LiteResidualModule.insert_lite_residual(net, 2, …, 5, 'relu', 2)`.

**S5. Disk.** `RunManager.train` writes a 35 MB `checkpoint.pth.tar` (including optimizer state) every epoch, plus a 19 MB `model_best` (`run_manager.py` lines 362-367). That is about 54 MB per run, or 6-8 GB over the campaign. Monkeypatch `run_manager.save_model` on the instance so it saves weights only at the last epoch; upstream stays unmodified.

**S6. Provenance.**
- Save `mem_repeat.py`'s output to JSON. The "agree within 1.1 MiB" claim and the 67.4 MiB steady-state FT-Last value appear in no saved file; `mem_harness_result.json` has FT-Last only as the first arm, at 82.9.
- Also record `nvidia-smi` utilization during the memory runs.

**S7. Wording.**
- Line 59: TinyTrain F.1 says inspecting 30-44% of layers is enough, i.e. the candidate set its Fisher selection scores, not the layers it trains (`tinytrain_paper.txt` lines 1905-1907). T3 is "the region TinyTrain searches over", which is still a fine prediction.
- Line 337: "must come from their protocol" → "may come from".
- Line 61: "label-dependent" conflates k with meta-training, domain and various-way protocol.

**S8. Budget details.**
- 17.3 h × 1.3 = 22.5 h, not about 21.
- O2 is underestimated. `run.py` passes no `validation_frequency`, so upstream validates on the 6,149-image test split every epoch: about +15 min per run, ≈ 3.7 h rather than 2.9.
- A 40%-busy shared GPU during training would break the run-time model; check utilization before each night.

**S9. Highest-value optional for the contradiction.** TinyTL-L (lite residuals only) with `--random_init_lite_residual`, batch 1 vs batch 8, 4 seeds (about 2.7 h). It is the closest available proxy for TinyTrain's TinyTL baseline (fresh adapters, no bias). "Catastrophic or not" is well powered even at n=4. Rank it above O4-O6 if B1(a) is adopted.

**S10. Zero-training memory ablation that ties memory to the batch story.** In the harness, replace every GN with a frozen per-channel affine (what eval-mode BN reduces to) and re-measure LB and B. Memory does not depend on weight values, so no checkpoint is needed. This shows directly how much of the 557 MiB is the price of GN, the very choice TinyTL made for batch-1 training.

**S11. Out-of-scope levers.** Name them in one sentence: activation recomputation (checkpointing), low-precision saved activations, and feature caching for FT-Last. Then "exactly two levers" (line 52) reads as a scoping choice, not a fact.

**S12. Reporting.** Always report d with its CI next to the label. At SD 1.8, P(retain | true d = −1) = 0.49, so "retains" will often hide a real 1 pp cost.

---

## What is good and should be kept

- **The unified memory harness.** The first-in-process workspace explanation (573.4 vs 557.5, 82.9 vs 67.4) is convincing and resolves an open item in context.md. The saved-tensor breakdown is the most valuable new fact in the proposal: it pinpoints the GN-input omission in TinyTL's profiler. Fix the framing (B3), keep the substance.
- **Memory from the harness, never from training runs.** The run peak is renamed to `run_peak_incl_eval_MiB`.
- **Final-epoch evaluation, with val used only for a frozen per-arm LR pilot on separate seeds.** This removes the few-shot checkpoint-selection problem without introducing test leakage, and is justified by the measured learning curves and the val-test correlation of only 0.169 within r1/r2.
- **Paired, seed-major design with shared k-shot subsets.** A partial block stays paired. `lr_choice.json` is frozen before block A. The G0/G0b gates and the smoke runs stay.
- **One tuned LR per arm from a common grid.** This is fairer than run.py's single 2e-4. The inheritance rules are explicit.
- **The budget arithmetic is internally consistent with the run-time model.** I reproduced 4.7 / 9.0 / 16.1 / 30.7 min per run and every block total. The old r2 runs (setup about 39 s, test about 31 s, 536-703 s of training with val every epoch) support the 1.0 min setup assumption.
- **Dropping non-contiguous lite-residual selection on memory grounds is sound.** The earliest trainable layer sets memory, and freezing every lite residual saves 20.5 MiB (3.7%). So is the "one dataset = one task, so task-adaptivity cannot be tested" argument, and dropping the resolution axis.
- **The device framing.** H2b asks which lever fits under a fixed measured budget. It is the right question and should stay.
- **The bash driver with skip-if-done, and the expected-outcome → conclusion table.** Keep the table; just apply the B2 relabeling.
