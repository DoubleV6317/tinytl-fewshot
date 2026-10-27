# Review of proposal_r3.md (round 3, final check)

**Verdict: MINOR REVISION.** Apply the two corrections below exactly as written; no further review round is needed. After that the design is ready to execute.

## R1-R3 from round 2: all resolved

- **R1 (engine bound).** `engine_bound_r3_result.json` matches Table A1 and every lever ratio:
  - LB/T3 2.60×, LB/T4 3.80×, LB/FT-Last 5.06×, batch 3.13×, FT-Full/LB 1.90×;
  - the profiler gap is 6.0× for LB and 6.1× for TinyTL-B.

  The "exact" profiler matches for FT-Last (30.8 MB) and for B with frozen-affine norms (31.7 MB at batch 8, 8.5 MB at batch 1) are real. They hold by construction, though, because the replay uses the profiler's own transient and mask conventions (see note 1).
- **R2 (H4 defined).** T2 runs unconditionally on Flowers (seeds 0-4) and Aircraft (seeds 0-5). H4 uses seeds 0-4 on both datasets, and the budgets' feasible sets are correct.
- **R3 (labels).** §4 now says "more than 2 pp" only for *inferior*. Table B1 reproduces `power_r3_result.txt` exactly for fixed n, and the H2 table matches `power_r3_h2_result.txt`.

**Accepted departures.**
- H2 uses n=5 with "catastrophic or not" as the primary rule. That is the claim TinyTrain actually makes, and it is resolved at 0.95 at both poles (SD 2.5). The weak "within 2 pp" (0.35) is stated as secondary.
- FT-Last on Aircraft uses 5 seeds. Its prior is −17.6 pp, so it resolves as *inferior* with probability 1.00.

## Internal consistency

Everything below checks out:
- Arms, seeds and the counts of 106 runs.
- LR inheritance: T4 and T2 take T3's LR, with no T4 pilot in the budget.
- The two conditionals, and the cut-order amounts (−0.53, −0.68 and −0.57 h).
- The fallback cost: the Aircraft hours total 9.08.
- The half-budget statistic (I re-derived 0.24-0.69 / 0.44-1.67 / 0.64-2.65 from the logs).
- The budget, recomputed from the block arithmetic: P-F 2.42 + P-Air 1.51 + A-F 6.03 + A-Air 4.86 + C-Air 2.71 + M 0.4 = **17.9 h core**, plus 0.62 + 1.08 = **19.6 h**.

The one inconsistency is the schedule (Must-fix 1).

## Must-fix

### 1. The schedule cannot make its only between-night decision

H2's LR-check trigger is d_b1 = LB-b1 − LB-b8 *on Aircraft*. LB-b8 on Aircraft is part of A-Air, which the schedule runs on Night 2, after the Day-2 decision.

**Exact correction:** replace the Night 1 / Day 2 / Night 2 rows of the summary schedule with:

| when | what | GPU-h |
|---|---|---|
| Night 1 | A-Air, then C-Air | 7.6 |
| Day 2 | H1-A and H2 labels, which decide the LR check | |
| Night 2 | A-F, then the H2 LR check if triggered | 6.0 + ≤ 1.1 |
| Day 3 | H1-F, H4 (both datasets), Pareto plots | |

This also balances the nights: 7.6 and 7.1 h instead of 8.7 and 6.0.

### 2. Summary "Why" item 1 says the ideal engine cannot fit 512 MB

246 MiB (258 MB) fits in 512 MB. Only the PyTorch figure does not.

**Exact correction:** replace "A 512 MB Pi Zero 2-class device (TinyTrain's target) cannot hold that." with:

> In PyTorch (558 MiB) the step does not fit a 512 MB Pi Zero 2-class device (TinyTrain's target); even an ideal engine needs about half of it.

## Non-blocking notes

1. **The profiler match is by construction.** In the summary and in Appendix A, replace "reproduces the profiler (exactly for TinyTL-B and FT-Last, …)" with:

   > reproduces the profiler (exactly, by construction, for FT-Last and for TinyTL-B with frozen-affine norms, since it uses the profiler's own transient and mask conventions; within 11% for LB). The gap with GroupNorm is therefore the GN inputs alone.

   FT-Last matches regardless of the norm type, so do not cite it as evidence about normalization.
2. **"The profiler does not see the depth lever at all"** (Appendix A) → "the profiler barely sees the depth lever (1.03-1.20×)".
3. **The §4 H2 row "a real loss of between |U| and 5 pp"** → "between |U| and |L| pp (|L| < 5)".
4. **The H2 LR check uses one seed and two alternative LRs.** With that much noise it will lean toward "LR-sensitive" (roughly 10-20% of the time when the drop is real). That errs in the conservative direction, but state it next to the rule. If an evening is free, use seeds 0-1 (+1.08 h).
5. **Fallback (Flowers only).** H4's "deeper on Aircraft" metric is undefined there, so report regret only. H2 on Flowers at n=6 uses the same rules.

## Execution

Nothing else would stop a 3-person team:
- about 1,100 lines of code over one calendar week;
- two GPU nights plus a daytime pilot;
- one human decision between nights;
- the Aircraft download, which needs the team's own approval and is already planned.
