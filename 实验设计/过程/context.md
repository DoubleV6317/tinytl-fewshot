# Shared context: TinyTL few-shot course project (for proposer and reviewer agents)

Everything below is measured data or verified source text. Treat it as facts to reason from, not as a plan.

## Paths

- Project root: `C:\Users\zhita\Documents\Codex\2026-09-10\this-class-will-feature-a-quarter\outputs\TinyTL`
  - Python venv: `.venv\Scripts\python.exe` (Python 3.12, PyTorch 2.5.1+cu124). Upstream code: `upstream\tinytl`, `upstream\once-for-all`.
  - `run.py` (launcher for the official trainer; hardcodes init_lr 2e-4 and label_smoothing 0.7 for every method).
  - `fewshot_resolution.py` (our few-shot trainer: k-shot sampling from the original Flowers102 train split via setid.mat, checkpoint chosen on the real val split, test used once, per-stage lite-residual pooling ratios with kernel sizes pinned).
  - `probe_methods.py` (memory probe of the six methods).
  - Reports: `复现结果-主方法.md`, `实验报告-分辨率分配.md`, `复现现状与后续计划.md`.
  - Per-run logs: `runs\fewshot-k5-*\logs\valid_console.txt` (one line per epoch with val top1 and train top1), `runs\fewshot-k5-*\result.json`.
- Team's latest story (Chinese, 3 pages, extracted text): `story.txt` in this folder.
- TinyTrain paper text (arXiv 2307.09988, extracted): `tinytrain_paper.txt` in this folder.
- Original proposal (Sept 18, older than the story): `...\outputs\proposal\proposal.tex`.
- Scratch scripts that produced the measurements below: `..\probe_depth.py`, `..\time_steps.py` (one level up from this folder).

## Constraints

- One RTX 3060 12 GB on Windows, jobs run serially; `n_worker=0` (Windows), so data loading dominates batch-8 runs.
- Three-person student team, course project. Deadline is not known; assume roughly 4 weeks remain. Aim for a core experiment that fits in about 20 GPU hours (two overnight runs) and at most about 40 GPU hours including extensions, plus at most a few days of coding.
- Only Flowers102 is downloaded. FGVC-Aircraft would be a few GB more.
- The only pretrained lite-residual checkpoint is `proxylessnas_mobile+lite_residual@imagenet@ws+gn` (GroupNorm + weight standardization). Upstream also has `--random_init_lite_residual`.

## Code facts (upstream TinyTL)

- `tinytl_fgvc_train.py`: all six methods (full, bn+last, last, tinytl-bias, tinytl-lite_residual, tinytl-lite_residual+bias) load the same ws+gn checkpoint with lite residuals inserted. In FT-Last / TinyTL-B / bn+last the pretrained lite residuals stay in the forward pass, frozen. FT-Full also trains them.
- Released scripts only cover TinyTL-L+B. `exp_scripts/batch8/flowers.sh`: 50 epochs, batch 8, Adam, cosine, init_lr 2e-4, label smoothing 0.7, frozen_param_bits 8. `exp_scripts/batch1/flowers.sh`: same with batch 1 and init_lr 3e-4. Other datasets use different learning rates (e.g. aircraft 16e-4, pets 0.5e-4).
- Upstream selects checkpoints on the test set (Flowers102 train+val = 2040 images for training, test 6149 used as "val").
- The TinyTL profiler: `memory_cost = peak_activation x batch + param_size` (peak, not sum). Frozen params counted at 8 bits, but 8-bit cluster quantization is stored in fp32 tensors in this code.
- `grouped_block_index` for ProxylessNAS-Mobile: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21]] (six stages). Block 0 and `first_conv` are the stem.

## Measured results

### Reproduction (official recipe, k=20 per class, seed 0)
TinyTL-L+B on Flowers102: final epoch 95.202%, best 95.219% (paper 95.5%). Profiler memory 37.126 MB (paper 37 MB). One seed only; none of the five baselines has been run yet.

### Six methods, profiler vs measured memory (batch 8, fp32, 224 px, isolated train steps, CUDA max_memory_allocated minus baseline)
| method | paper MB | profiler MB | measured MiB |
|---|---|---|---|
| FT-Last | 31 | 31.6 | 82.9 |
| FT-Norm+Last | 192 | 192.4 | 538.9 |
| TinyTL-B | 32 | 31.7 | 538.7 |
| TinyTL-L | 37 | 37.1 | 397.5 |
| TinyTL-L+B | 37 | 37.1 | 559.5 |
| FT-Full | 391 | 390.8 | 598.8 |
Full / L+B: profiler 10.53x, measured 1.07x. At batch 1: profiler 4.69x, measured 1.30x; measured MiB at batch 1: Last 25.1, Norm+Last 101.3, B 101.1, L 101.2, L+B 121.8, Full 158.4.

### Truncated backprop: L+B updated only from stage s onward (bias + lite residual of earlier layers frozen), batch 8
| variant | profiler MB | measured MiB | trainable params |
|---|---|---|---|
| L+B full (from stem) | 37.1 | 573.4 | 1,962,310 |
| B, skip first_conv | 31.7 | 498.7 | 149,590 |
| B, skip first_conv + block 0 | 31.7 | 448.5 | 149,542 |
| L+B from stage 0 (block 1) | 37.1 | 471.4 | 1,962,230 |
| L+B from stage 1 (block 5) | 37.1 | 328.7 | 1,944,182 |
| L+B from stage 2 (block 9) | 36.8 | 226.5 | 1,863,766 |
| L+B from stage 3 (block 13) | 36.0 | 168.6 | 1,578,246 |
| L+B from stage 4 (block 17) | 34.6 | 105.2 | 1,113,126 |
| L+B from stage 5 (block 21) | 32.3 | 69.2 | 362,598 |
| feature_mix + head only | 31.6 | 67.4 | 131,942 |
Measured memory is set by the earliest layer that needs a gradient, not by the number of trainable parameters. Freezing all lite residuals in full L+B saves only 20.8 MiB (3.6%). Note: two measurement scripts gave 559.5 vs 573.4 MiB for the same L+B config; there is no single unified memory harness yet.

### GPU time per training step (synthetic data, no data loading)
| variant | batch 1 | batch 8 |
|---|---|---|
| L+B full | 58.6 ms/step | 57.4 ms/step (7.2 ms/image) |
| L+B from stage 3 | 39.7 ms/step | 40.3 ms/step (5.0 ms/image) |
| head only | 25.0 ms/step | 24.8 ms/step (3.1 ms/image) |
Step time is launch-overhead bound: batch 1 costs the same per step as batch 8, i.e. 8x more GPU time per image.

### Run-time model from completed runs (L+B, batch 8, n_worker=0)
About 16.8 ms per training image (including data loading) and about 3 ms per eval image. 50 epochs at batch 8: k=2 about 3 min, k=5 about 7 min, k=10 about 14 min, k=20 about 29 min (+ about 0.3 min for one test pass over 6149 images, + about 2.5 min if validating 1020 images every epoch). Batch 1 is roughly 4x slower overall (about 68 ms per image). Weight quantization (k-means) adds some fixed setup time per run. These are extrapolations, about +/-30%.

### Few-shot k=5 resolution experiment (TinyTL-L+B, 3 seeds, val split for checkpoint, test once)
Test top-1: r1 84.21 +/- 0.82, r2 (official) 82.99 +/- 1.10, r4 75.06 +/- 0.84. Paired r1-r2: +1.22 (seeds +0.85 / -0.39 / +3.20), SD 1.83, p=0.367. Paired r4-r2: -7.93, p=0.005. With 3 seeds the minimum detectable effect at 80% power is about 2.95 pp; about 18 seeds would be needed for +1.22 pp. Val-test correlation is 0.979 over all 9 runs but only +0.169 within the r1/r2 runs; val picked the wrong config in 1 of 3 seeds. Full details: `实验报告-分辨率分配.md`.

### Learning curves at k=5 (r2, three seeds)
Train top-1 reaches about 99-100% by epoch 10-15. Val top-1 keeps rising slowly or plateaus (e.g. seed 0: ep15 86.2, ep30 86.7, ep50 87.3); it never falls back. Val loss bottoms around epoch 15 (about 2.31) and is about 0.03 higher at epoch 50. A large train-test gap exists, but more training does not reduce accuracy.

## Verified facts from the TinyTrain paper (tinytrain_paper.txt)

- Table 1, ProxylessNAS row, Flower column: None 80.5, FullTrain 86.3, LastLayer 85.5, TinyTL 86.2, SparseUpdate 87.8, TinyTrain 92.7. ProxylessNAS averages over nine domains: FullTrain 65.0, LastLayer 57.0, TinyTL 63.7, TinyTrain 68.3. MobileNetV2 Flower: LastLayer 86.3, TinyTL 85.9, FullTrain 84.4.
- Table 2: TinyTL backward memory 541-587 MB, FullTrain 857-1,049 MB. The text says a batch size of 100 was used for these two baselines because their accuracy "degrades catastrophically with smaller batch sizes"; LastLayer, SparseUpdate and TinyTrain use batch 1. The text says the TinyTL and FullTrain figures exceed the RAM of IoT devices such as Pi Zero (512 MB).
- In TinyTrain the TinyTL baseline "updates the augmented lite-residual modules while freezing the backbone" (no mention of bias or of GroupNorm).
- Protocol: backbones ImageNet-pretrained, then ProtoNet meta-training on MiniImageNet. Meta-test on nine Meta-Dataset domains, 200 tasks per dataset, various-way various-shot; end-to-end training uses about 25 samples for about 40 iterations. Devices: Pi Zero 2 and Jetson Nano.
- Meta-training raises the accuracy of every method by 0.6-31.8 pp, and TinyTrain's by 5.6 pp on average, so the Table 1 baselines also sit on meta-trained backbones.
- Layer importance uses the Fisher potential of activations, computed once per target task from the given samples; layers are scored by Fisher potential per parameter and per MAC, then chosen greedily under memory and compute budgets (e.g. about 1 MB backward memory, 15% of MACs). Then the top-K channels in each layer are chosen.
- Dynamic channel selection beats static channel selection by 0.8-1.7 pp (vs random) and 1.9-2.5 pp (vs L2-norm). Their layer selection beats L2-norm top-k layer selection by up to 2.0 / 5.1 / 9.2 pp (MCUNet / MobileNetV2 / ProxylessNAS).
- Figure 3 (MCUNet, Traffic Sign): accuracy gain per layer peaks at the first (pointwise) layer of each block. The main text does not appear to say that important layers lie mostly in the later part of the network (the story claims this; it may come from an appendix).

## Open questions already raised about the story

The story proposes (a) task-adaptive selection of which blocks train lite residuals, (b) imprinting to initialize the classifier head, (c) a batch-size study to resolve the TinyTL-vs-TinyTrain contradiction about small batches, across several k. Its memory narrative still uses the profiler's 37 vs 391 MB. Treat the facts above as evidence that bears on each of these; do not assume any of them is settled either way.
