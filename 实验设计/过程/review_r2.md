# Review of proposal_r2.md (round 2)

**Verdict: MINOR REVISION.**

The revision answers round 1 in substance:
- Aircraft is now in the core, with TinyTrain Table 6 as pre-registered priors.
- Decision rules use four labels, 93% CIs and Holm.
- Memory is reported in three accountings, plus a GN ablation.
- Batch 1 uses its published LR, with an LR check before any negative claim.
- A convergence check exists that can actually fail.
- Fisher selection is in the core.

Three things remain, and each is cheap:
1. The engine bound omits the forward transient, which inflates one headline ratio.
2. H4 (Fisher) is undefined on Flowers at the 256 MiB budget, because the T2 arm it may pick is not run there.
3. Two outcome-table rows still read "not retained" as "costs more than 2 pp".

None of these needs a redesign. I also recommend simplifying the conditional logic for a 3-person team (section 3).

What I checked:
- I read every new script and result file.
- I recomputed all block budgets from the per-run minutes.
- I re-derived the half-budget numbers from the old run logs.
- I read upstream `batch1/*.sh`, `batch8/*.sh` and `make_aircraft.py`.
- I ran one CPU-only check of the forward transient, `design\review_r2_transient.py` (no training).

---

## 1. Status of B1-B6

| # | status | evidence |
|---|---|---|
| B1 prior work / dataset | **Resolved** | Table 6 is cited and the novelty sentence withdrawn. The five differences from Table 6 are listed. The AdapterDrop rungs are mapped to T2-T4 by the share of adapter blocks kept: T3 keeps 9/21 = 43%, about AD-50%; T4 keeps 5/21, about AD-75%. The priors match `tinytrain_paper.txt` lines 1233-1239: Flower −0.6/−1.0/−1.9/−0.7, Aircraft −2.1/−3.1/−10.6/−17.6. Aircraft is in the core, with a G-data fallback. The LR pilot is centred on upstream's 16e-4 (`batch8/aircraft.sh`). Implementation notes are in N4-N5. |
| B2 decision rules | **Partly resolved** | The four labels, 93% CIs and Holm code are correct (checked in §2.2). Two problems remain (R3): two §8 rows over-read "size unresolved", and the Aircraft label table assumes an extension that is not in the core. |
| B3 memory narrative | **Partly resolved** | The three accountings and the GN → frozen-affine ablation are exactly what was asked, and the "GN is what the profiler leaves out" finding is now well supported. But the engine bound leaves out the forward transient. That doubles FT-Last's bound and halves the depth ratio quoted in §1 and in the lever table (R1). |
| B4 batch-1 LR | **Resolved** | LB-b1 uses the published batch-1 LR per dataset. The table of upstream ratios (Flowers 1.5, Aircraft 0.19, Car 0.25) is correct; I checked the `.sh` files. The LR check (÷3 and ×4, seeds 0-1, "removes more than half of Δ on val") is triggered by every negative label, including "size unresolved". The scope sentence about TinyTrain's condition is right. |
| B5 convergence / H3 | **Resolved** | H3 is now optional, and O2 carries the step-matched control. The half-budget check can fail: on the old r4 runs, seeds 0-1 give a mean of 1.62 > 1.5. It is noisy, though (N8). |
| B6 sweep wording / Fisher | **Partly resolved** | The "exhaustive" and "upper bound" wording is deleted. Fisher-under-budget is core, with a correct ProtoNet-equivalent head (2μ_c·f − ‖μ_c‖² equals −‖f − μ_c‖² up to a per-sample constant). I accept the rebuttal that T2 should stay out of the Flowers Holm family. But H4 needs T2 accuracy on Flowers whether or not T2 is in Holm (R2). |

---

## 2. Checks of the new numbers and scripts

### 2.1 Engine bound and GN ablation (`engine_bound.py`, `engine_bound_result.json`)

The saver-tagging logic is sound:
- a storage saved by a norm layer, or by a conv/linear with a trainable weight, is kept in full;
- a storage whose savers are only activations (or activations plus frozen convs, as with in-place ReLU6 outputs) is kept at 1 bit;
- a storage saved only by frozen convs, average pooling or shape-only ops is dropped.

The breakdown is consistent with round 1: LB keeps 195.2 + 4.5 MiB of saved tensors, B keeps 172.8 + 4.6. In PyTorch, GN costs 174.6 MiB at batch 8 (LB 559.8 → 385.2 with GN replaced).

**The problem is that transients are excluded.** The proposal says the forward peak "adds at most about 26 MiB at batch 8". That is true, but for an arm that saves almost nothing, those 26 MiB are the whole activation cost. I measured the inference transient with TinyTL's own `count_activation_size(require_backward=False)` and took max(saved, transient) + weights/optimizer/input (`review_r2_transient.py`):

| arm (batch 8) | engine bound (r2) | transient | max-bound | measured |
|---|---|---|---|---|
| FT-Last | 24.2 | 24.5 | **48.7** | 67.4 |
| T4 | 63.4 | 24.5 | 63.4 | 105.2 |
| T3 | 93.4 | 24.5 | 93.4 | 169.4 |
| LB | 244.9 | 24.5 | 244.9 | 559.8 |
| B | 201.8 | 24.5 | 201.8 | 537.1 |

Consequences:
- Under the engine, the LB / FT-Last ratio is **5.0×, not 10.1×**. That affects the lever table, §1 ("cut memory by 2.6-10×"), and the §8 FT-Last row ("10.1× under the engine").
- The depth ratios LB/T3 (2.6×) and LB/T4 (3.9×), and the batch ratio (3.1×), are unchanged.
- The claim that B with frozen-affine norms at 29.4 MiB "matches the profiler's 31.7 MB" is a coincidence. The engine figure has fp32 weights and no transient; the profiler has 8-bit frozen weights and includes the transient. On the profiler's own basis (transient included, frozen weights at 8 bit), B with frozen affine comes to about 35 MB. That still supports "the profiler is roughly right only for frozen-statistics norms"; it just should not be called a match.

### 2.2 Holm rules and their Monte Carlo (`power_r2.py`, `power_r2b.py`)

What is correct:
- The NI p-value is 1 − F_t((m + 2)/SE).
- Holm is step-down with thresholds 0.035/3, 0.035/2 and 0.035.
- The paired model shares LB's error across arms (correlation 0.5).
- The type-I rate of "retains" at true d = −2 is 4.2-5.9% across the designs. "About 5%" is a fair description.
- The H1-F prior table matches the output: T3 retains 32% plus 22% "unadjusted only"; T4 is size-unresolved 91%; FT-Last retains 47% plus 27%.

Two gaps:
- **The Aircraft H1 MC simulates an extension from n=5 to n=10, but the Aircraft extension is optional (O1).** Without it, the probabilities differ. At the T3 prior (d = −3, SD 2.5), P(inconclusive at n=5) is **0.38**; the table reports inconclusive 0.03 and size unresolved 0.79. At SD 3.5 it is 0.60. The table must show the core design.
- **Holm controls only the "retains" claims.** Inferiority claims (U < −2) are unadjusted. That is acceptable, but say so.

### 2.3 Aircraft data preparation (§5)

What is right:
- Using the official `images_variant_{train,val,test}.txt` (3,334 / 3,333 / 3,333, 100 variants) is correct, and better than upstream's hanlab lists.
- Cropping the 20-px banner is standard practice.
- Pre-resizing to a shorter side of 512 px is well argued. `decode_time.py` measures 10.3 vs 17.3 ms per image with jitter, and the smallest crop of about 290 px is still a downsample.
- The val subset (10 per variant, `Random(12345)`) and k-shot sampling from train both mirror Flowers.
- The LR grid {4e-4, 1.6e-3, 6.4e-3} plus the edge rule is centred correctly.

Implementation notes are N4-N5. Neither blocks the design.

### 2.4 Batch-1 rule and whether H2 on Aircraft is well posed

H2 on Aircraft is well posed. It tests TinyTL's published batch-1 recipe in the domain where the batch lever matters for H2b, and TinyTL's own batch-1 claim covers Aircraft.

The comparison is asymmetric: LB-b8 gets a pilot-tuned LR, while LB-b1 uses the published 3e-4.
- That makes "no meaningful degradation" a conservative conclusion.
- It makes negative labels LR-suspect, which is exactly what the LR check is for. State the asymmetry in §3.

Power for "no degradation" is weak (0.61 at d = 0, SD 2.5, n 5→8), and the proposal says so. "Catastrophic or not" is well resolved (0.95 at d = −10).

H2b is close to predetermined if the priors hold (e ≈ +10 → P(batch better) = 1.00). That is fine for a decision rule, but call it a confirmation, not a discovery.

### 2.5 Convergence check

Δ_half = val(50) − val(25) can fail, but it is noisy. Per-seed values on the old runs:
- r2: +0.29 / +1.67 / +0.10;
- r1: −0.30 / +0.88 / +0.69;
- r4: +2.16 / +1.08 / −0.98.

The 2-seed r4 mean is 1.62 on seeds 0-1 but 0.05 on seeds 1-2. Aircraft uses one pilot seed. Expect one or two false triggers across six arm × dataset cells. Each trigger costs only a 100-epoch run of about 0.5 h, and the consequence is a label ("at the official budget"), so the stakes are low. See N8 for a noise reduction.

### 2.6 Budget

I reproduced every block from the per-run minutes:

| block | GPU-h |
|---|---|
| P-F | 2.15 |
| P-Air | 1.67 |
| A-F | 5.32 |
| A-Air | 3.31 |
| C-Air | 2.71 |
| Air-follow | 0.87 |
| M + Fisher | 0.25 |
| **core** | **16.3** |
| F-ext | +1.4 |
| C-ext | +2.2 |
| **core + likely conditionals** | **19.8** |

The arithmetic is right.

Omissions, about +0.5-1 h:
- the T4 LR pilots when T3 and FT-Last disagree: 2 runs on Aircraft, which is likely because FT-Last will want a higher LR, and 4 on Flowers;
- smoke runs;
- R2 adds 0.7 h.

With +30% on the run-time model, the likely total is about 26-27 h. That is within the 40 h ceiling but above the 20 h core target. The pre-registered cut order handles it.

---

## 3. Feasibility and course-project fit

**Is the question clear and worth answering?** Yes. "How shallow can few-shot TinyTL back-propagate in a near and a far domain, and is batch 1 the better way to fit 128 MiB?" is a crisp question:
- it is anchored to a published prior (Table 6);
- it has a device-relevant decision (H2b);
- it adds a mechanistic memory finding (GN is what the profiler omits).

It is clearly more than a parameter sweep, and it engages both papers and the batch contradiction. One thing changed from the story: the story's k-axis ⑧ has been replaced by a domain axis. The team should rewrite story ⑧ to match, and quote the TinyTL paper's full-data gaps (Flowers 95.5 vs 90.1, and the Aircraft equivalents) as a qualitative "more data" anchor.

**Can 3 people build and run it in about 4 weeks?** Yes, with one caveat.
- About 1,100 lines of new code is fine for three people. The "2-2.5 person-days" estimate is optimistic; plan about one calendar week, including smoke tests.
- The GPU plan is three nights.
- The real risk is control logic. r2 has about ten decision points that need a human between nights:
  - G-data, G0, G-time, G1, G2;
  - the F-ext and C-ext looks;
  - the H2-LR check;
  - the half-budget checks;
  - F-T2 / F-T5;
  - the Aircraft follow-up arm;
  - pilot edge extensions (up to two per arm);
  - T4/T2 LR inheritance with a pilot fallback;
  - a four-step cut order.

  Each is reasonable alone, but together they are where a student team will make bookkeeping errors, and they make the report hard to write.

**Recommended simplification** (strongly advised, not blocking; about the same GPU time, deterministic):
1. **Fixed sample sizes, no extension looks.** Flowers n=10, Aircraft n=6 for H1, H2 n=8 (LB-b1 and LB-b8 at seeds 0-7). This removes F-ext, C-ext and their type-I inflation. Inconclusive arms are simply reported as such. Cost: Aircraft 6th seed about 0.7 h, plus H2 at n=8 about 2.2 h (equal to today's "likely" C-ext). Total about 20 h.
2. **T2 unconditional on both datasets**, as an exploratory arm outside Holm (this is also R2). Drop F-T2, F-T5 and the Aircraft follow-up rule. T5 adds 1 MiB over FT-Last and is not device-relevant.
3. **Fixed LR inheritance.** T4 and T2 always take T3's LR; drop the "if T3 and FT-Last agree" branches.
4. **Keep only two conditionals:** the H2 LR check (essential for B4) and the half-budget 100-epoch check.

---

## 4. Remaining blocking issues

### R1. Include the forward transient in the engine bound, and restate the ratios that depend on it.

Required:
- Define the engine bound as max over the forward pass of (needed saved-so-far + live transient), plus weights and optimizer state. The simplest valid lower bound is max(saved, transient) + extra, as in `review_r2_transient.py`.
- Equivalently, run TinyTL's profiler algorithm with GN inputs charged wherever gradients pass through. That puts the result on the profiler's own basis, so the gap is attributable to GN alone.
- Update:
  - FT-Last (b8 ≈ 49 MiB);
  - the lever table (LB/FT-Last under the engine ≈ 5×);
  - §1 ("2.6-5×");
  - §8 ("10.1×");
  - the B-with-frozen-affine vs profiler sentence (compare on the same basis, about 35 vs 31.7 MB).

### R2. Make H4 well defined: T2 accuracy is needed on both datasets.

At the 256 MiB budget:
- Fisher's prefix-greedy rule can return {G2, G3, G45}, i.e. T2 at 226 MiB.
- The static rule "deepest feasible" is T2 by definition.

On Flowers, T2 runs only if T3 is *inferior*, which the prior makes unlikely. On Aircraft it runs only if T3 does not retain. So the regret of both Fisher and the static rule is undefined in the likely case.

Restricting Flowers to the run arms would instead bias the pre-registered "deeper on Aircraft" metric, because Aircraft would have the deeper option and Flowers would not.

Required:
- Run T2 unconditionally on both datasets as an exploratory arm outside Holm. On Flowers, seeds 0-4 suffice (5 runs, about 0.7 h).
- Evaluate H4 on seeds 0-4 on both datasets, which also balances the cross-dataset comparison.
- Define the oracle as the best run arm within the budget.

### R3. Finish the B2 relabeling.

- **Two §8 rows over-read "size unresolved".**
  - The row "H1-A: … T3 size unresolved or inferior, T2 retains → the last 3 stages are not enough" should say "not enough" only if T3 is *inferior*.
  - The row "H1-A: T2 also not retained → … before costing > 2 pp" may say "> 2 pp" only if T2 is *inferior*. Otherwise: "costs between 0 and |L| pp; full depth not shown necessary".
- **Recompute the Aircraft label table for the core design (n=5, no extension).** For example, T3 is inconclusive 38% at SD 2.5. Alternatively, adopt the fixed n=6 of §3.
- **State that Holm adjusts only the "retains" claims.**

---

## 5. Non-blocking notes

- **N1 (engine accounting).** "Full fp32 if a norm saves it" is an fp32 ideal. Say in one line that lower-precision saved activations or recomputation would lower it further (S11 is already listed as out of scope).
- **N2 (Aircraft seed allocation).** T4 and FT-Last resolve trivially on Aircraft (priors −10.6 and −17.6). The open questions are T3 and T2 (−2.1 to −3.1). If any budget frees up, spend it on LB/T2/T3 seeds on Aircraft, not on O-blocks.
- **N3 (H2 asymmetry).** b8 is pilot-tuned and b1 is not. Say this beside the H2 labels, and note that it makes "no degradation" conservative.
- **N4 (Aircraft parsing and folders).** Lines in `images_variant_*.txt` are `<id> <variant>`, and variant names contain spaces ("Cessna 172", "Boeing 717"). Split on the first space only. Two variants contain "/" ("F-16A/B", "F/A-18"); as folder names they silently nest into classes "F-16A" and "F". Use the integer index from `variants.txt` as the folder name. Assert 100 classes and identical class lists in train, val, test and every k-shot root.
- **N5 (protocol deviation).** Upstream `make_aircraft.py` neither crops the banner nor uses the official 3-way split. Both are deliberate deviations from "TinyTL's own recipe"; list them with the Flowers deviations.
- **N6 (budget line items).** Add the likely T4 LR pilot runs (Aircraft 2, Flowers 4) and the smoke runs to the core table.
- **N7 (Fisher head).** With k=5, each support image is a fifth of its own prototype, so the ProtoNet loss on the support set is artificially small, and the Fisher sums are dominated by a few samples. Use leave-one-out prototypes for each sample's own class, and plain CE without label smoothing. Log the support loss so a degenerate Fisher is visible.
- **N8 (half-budget noise).** Compare the mean of val at epochs {45, 50} with the mean at {20, 25}; both are already logged, since val runs every 5 epochs. Apply the check to T4/T2 by inheritance, or note that it is not applied.
- **N9 (H4 oracle).** Report regret both against the per-seed oracle and against the arm with the best mean. A per-seed oracle includes test noise and raises every rule's regret.
- **N10 (provenance).** Keep the r2 helper scripts with the project's memory tooling. `measure_memory.py` should emit the transient-inclusive engine column (R1).

---

## 6. What must be kept

- **Aircraft in the core**, with Table 6 rung priors and a pre-registered fallback to Flowers-only. This turns the Flowers-only risk into a two-domain contrast with a falsifiable prediction ("opposite ends").
- **The four-label scheme, 93% CIs and Holm over {T3, T4, FT-Last}.** Dropping the monotone fixed sequence is right, given the non-monotone Flower prior.
- **The three memory accountings, the saved-tensor tagging, and the GN → frozen-affine ablation.** "The GN that enables batch 1 is what TinyTL's memory model omits" is the project's strongest single finding. Fix only the transient (R1).
- **Batch 1 at the dataset's published LR, with the pre-registered LR check** before any negative H2 claim, and the honest scope statement about TinyTrain's condition.
- **Analytic backward MACs** in place of GPU step times, and memory plus MACs for the H2b tie-break.
- **RNG reseeding before `train()`, the KMeans `random_state` patch, final-epoch-only checkpoints, the profiler batch and `require_backward` fixes, and the corrected log parsing.**
- **The step-matched control** attached to the optional H3.
- **The pre-registered cut order and the G-data / G-time gates.**
