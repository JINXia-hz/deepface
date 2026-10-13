# deepface — Fully Convolutional Heatmap Face Detection

Detect all faces in any image. The default detector is a ResUNet dense score-map model; the original CNN binary classifier (36×36 grayscale input, face/non-face output) with its fully-convolutional heatmap pipeline is kept for comparison.

All shared logic lives in the `deepface/` package; runnable scripts live in `tools/` and are invoked as modules from the repo root (`py -m tools.<name> ...`), so no path hacks are needed anywhere.

## Directory Structure

```
├── deepface/           # Package: all shared logic (no side effects on import)
│   ├── models.py       # Net (36×36 window classifier) + ResUNet (dense score map) + load_model()
│   ├── data.py         # Crop dataset loading (preloaded into memory, train/valid/test DataLoaders)
│   ├── wider.py        # WIDER FACE annotation parsing, scene tile samplers, stride-8/stride-1 labels
│   ├── detect.py       # Pyramid scanning for both backends, weighted/cluster NMS, scale refinement, box calibration, detect()
│   └── evaluate.py     # Crop metrics, eval job lists, WIDER held-out recall, official WIDER AP protocol
├── tools/              # Executable scripts (run from repo root: py -m tools.<name>)
│   ├── detect.py           # Detect one image: py -m tools.detect <img> [model] [thr]
│   ├── train_cnn.py        # Train the small CNN: py -m tools.train_cnn <epochs>
│   ├── test_cnn.py         # Evaluate the small CNN on the test set
│   ├── export_mistakes.py  # Export a grid of misclassified test samples to outputs/
│   ├── finetune_fcn.py     # FCN fine-tuning: <steps> <lambda> <out> <seed>
│   ├── train_deep.py       # ResUNet training from scratch: <steps> <lambda> <out> <seed>
│   ├── eval_fcn.py         # Full evaluation (test crops / holdout FPs / samples / WIDER recall)
│   ├── cache_candidates.py # Cache low-threshold candidates: <model> <out.pt>
│   ├── sweep.py            # Threshold × NMS sweep on cached candidates
│   ├── measure_kernel.py   # Importance kernel / response envelope / GT face spacing
│   ├── eval_wider.py       # Official WIDER val: scan (multiprocess, caches raw candidates) + per-tier AP
│   ├── calibrate_box.py    # Box shape calibration grid search on cached detections
│   ├── level_redundancy.py # Cross-level redundancy: level occupancy, thinning AP
│   ├── level_support.py    # Cross-level support as TP/FP signal (refuted)
│   ├── scale_profile.py    # Scale-space profile shapes, TP vs FP (real but redundant)
│   └── scale_interp.py     # Sub-grid scale interpolation on a thinned ladder
├── models/             # Trained model weights
│   ├── model.pth       # Legacy model (trained with per-window histogram equalization, test 98.77%)
│   ├── model_noeq.pth  # Baseline (no equalization, test 97.93%)
│   ├── model_fcn.pth   # FCN fine-tuned on WIDER val scenes
│   ├── model_fcn2.pth  # FCN fine-tuned on WIDER train+val scenes, 3000 steps
│   └── model_deep.pth  # ResUNet default detector (v1 2000 steps + 4000-step continuation)
├── data/               # Scene data (git-ignored): COCO val2017, no_person, holdout, widerface
├── train_images/       # Training set (0=non-face, 1=face, 36×36); git-ignored, restored from crops.tar.gz
├── test_images/        # Test set; git-ignored, restored from crops.tar.gz
├── crops.tar.gz        # Packaged crop dataset (train_images/ + test_images/) — download-and-use, no external data hunt
├── samples/            # Sample images to detect on
└── outputs/            # Detection outputs (<name>_<n>faces.png), cache/ for candidate caches
```

## Setup

```bash
pip install -r requirements.txt   # runtime deps: torch, torchvision, numpy, scipy, pillow
tar -xzf crops.tar.gz             # restore the crop dataset (train_images/ + test_images/)
```

`crops.tar.gz` (~110 MB) is committed in the repo root via **Git LFS** and packages the full 36×36 grayscale crop dataset — `train_images/` (64771 faces, 26951 non-faces) and `test_images/` (798 / 6832) — so the repo works right after download. If `crops.tar.gz` shows up as a tiny text file after cloning, install Git LFS and run `git lfs pull` to fetch the real archive. The two directories are git-ignored and restored by extracting the archive. Scene data under `data/` (COCO val2017, no_person, holdout, WIDER FACE) is not included; download it separately to run the scene-level tools (`train_deep`, `eval_wider`, ...).

For development, `pip install -r requirements-dev.txt` (ruff + mypy); CI runs `ruff check deepface tools` and `mypy deepface tools` on every push/PR.

## Usage

```bash
py -m tools.detect samples/test.jpg                       # ResUNet detector (default model_deep.pth, thr 0.85)
py -m tools.detect samples/test.jpg models/model_fcn.pth  # Net heatmap backend (auto-detected from weights)
py -m tools.train_cnn 10    # Train the small CNN, saves models/model_noeq.pth
py -m tools.test_cnn        # Evaluate the small CNN on the test set
py -m tools.export_mistakes # Export misclassified samples
```

`load_model()` infers the backend from the state dict keys, so every tool (`detect`, `eval_fcn`, `cache_candidates`) accepts either model without extra flags.

## Detection Principle: From Sliding Window to Fully Convolutional Heatmap

The traditional approach is sliding window: downscale the image through a pyramid, and at each level crop 36×36 windows with stride 8 and classify them one by one. Most of the computation is wasted on overlapping regions — adjacent windows share 28/36 of their pixels, so convolutions are recomputed many times.

Key observation: **convolution is itself a sliding window**. Feed the whole image into the network directly:

1. `net.features(whole image)` — the conv layers convolve the entire image, equivalent to convolving every 36×36 window separately, but shared regions are computed only once. Outputs a feature map `[1, 64, H', W']`.
2. `F.avg_pool2d(f, kernel_size=4, stride=1)` — equivalent to the per-window global average pooling `mean(dim=(2,3))` in the original forward pass.
3. `F.conv2d(f, fc.weight.view(2,64,1,1), fc.bias)`
4. After softmax, take the class-1 channel to get the heatmap: each position is the face probability of the corresponding window.

Coordinate mapping: the network contains three 2×2 maxpools, so heatmap pixel (a, b) corresponds exactly to window top-left corner (8b, 8a) in the pyramid level; multiply by the current pyramid level's scale to map back to original image coordinates. Boxes are drawn after threshold filtering and NMS (IoU 0.2).

## Runtime Comparison (test.jpg 1000×667)

| Approach | Time |
|---|---|
| Sliding window + per-window PIL histogram equalization | 617 s |
| Sliding window + vectorized equalization (batched inference) | 18.8 s |
| **Fully convolutional heatmap (current approach)** | **~5 s (including ~3 s interpreter/torch startup)** |

The sliding-window approach (the old `detect.py`) has been retired and deleted. On large images (6000×4000) it required batched inference to avoid a 30GB memory peak, whereas the heatmap approach is naturally memory-light and completes in a single forward pass over the whole image.

## Failed Experiment: Hard Negative Mining

We tried mining hard negatives: filter COCO val2017 down to person-free images (2693 of 5000 contain people — filtering is mandatory), run the detector at a low threshold (0.8), and add the false-positive crops to `train_images/0/`. Two rounds were trained and compared against the original model and a plain-retrain control:

| | original | mined v1 | control | mined v2 |
|---|---|---|---|---|
| test accuracy | 97.93% | 96.92% | 98.66% | 93.39% |
| TP / 697 faces | 643 | 574 | 720 | 297 |
| samples detections | 1/2/32/1 | 6/7/27/3 | 2/10/23/2 | 0/2/9/6 |

Both mined versions collapsed recall (v2 lost half the test faces and missed the face in `standard_beatrice-cenci.jpg` entirely). Root causes identified:

1. Face-like textures (dot lattices, fur, dials) overlap with weak-evidence real faces in feature space — pushing the decision boundary against them necessarily sacrifices borderline real faces.
2. A heatmap score is not comparable to a standalone 36×36 forward pass: each conv layer's padding gives the fully-convolutional window ~15px of real surrounding context that a cropped 36×36 input never sees (same pixels, FCN score 0.9978 vs direct score 0.0149). Mining was fixed to re-score crops with a direct forward pass, but v2 showed the mechanism itself is harmful regardless of sample fidelity.

Also note the large retrain variance (control FN 77 vs original FN 154 on identical data) — single-run comparisons on a 697-face test set are noisy.

Conclusion: this direction is abandoned; the original model is kept. A viable alternative, if FP reduction is ever revisited, is FCN-mode fine-tuning — penalize heatmap responses directly on face-free scene images so training matches the deployment computation exactly.

## FCN Fine-tuning on Scene Data (WIDER FACE)

Following the conclusion above, the baseline was fine-tuned in fully-convolutional mode on WIDER FACE scene tiles (192×192, stride-8 heatmap labels, positive = 36px window with IoU ≥ 0.5 against a face scaled to 26–52px), mixed with the original crop dataset as an anchor against catastrophic forgetting. Data scaling (val-only → train+val, 1000 → 3000 steps) drove false positives down but recall stayed flat:

| model | training data | holdout FPs (100 imgs) | WIDER recall@0.995 | test crops |
|---|---|---|---|---|
| model_noeq | — | 259 | 0.477 | 97.93% |
| model_fcn | WIDER val, 1000 steps | 61 | 0.493 | 98.34% |
| model_fcn2 | WIDER train+val, 3000 steps | 44 | 0.485 | 98.16% |

Recall not responding to a 4× data increase suggested the bottleneck had moved from data to model capacity — motivating the ResUNet experiment below.

## ResUNet Dense Detector (current default)

`deepface/models.py` also hosts a small ResUNet (549k params vs 72k): two maxpool stages with residual blocks, a bottleneck at stride 4, and a decoder with skip connections back to full input resolution, ending in a 1×1 conv that emits a per-pixel face score map (stride 1 instead of stride 8).

Trained from scratch (`tools/train_deep.py`, 2000 Adam steps on the same WIDER tiles) with a label scheme adapted to dense output: a radius-6 disk at each in-range face center is positive, the disk inside every plausible face box is ignore, everything else is negative; focal loss with separate positive/negative means; the original crop dataset still mixed in as an anchor. Detections are 3×3 local maxima on the sigmoid score map, mapped to 36×scale boxes centered on the peak, merged across the pyramid with the same weighted NMS.

Result at matched false positives — the capacity hypothesis confirmed:

| model | holdout FPs | WIDER recall |
|---|---|---|
| model_fcn2 @ 0.995 | 44 | 0.485 |
| model_deep @ 0.8 | 39 | **0.617** |

Default threshold 0.85 (`deepface/detect.py` BACKENDS) trades some recall for clean output on non-photographic images: 16 holdout FPs, recall 0.542, samples 1/1/17/1. Cost: ~8 s/image on CPU (about an order of magnitude slower than the tiny Net) and ~2.8 h of CPU training.

## Official WIDER FACE Val Evaluation

Full official protocol (`tools/eval_wider.py`, a faithful port of the eval_tools MATLAB code: greedy matching at IoU 0.5, off-tier matches ignored, global min-max score normalization, VOC all-point AP) over all 3226 val images, Easy/Medium/Hard tiers (cumulative):

| tier | AP | precision@0.85 | recall@0.85 |
|---|---|---|---|
| Easy | 0.559 | 0.60 | 0.58 |
| Medium | 0.536 | 0.71 | 0.51 |
| Hard | 0.231 | 0.71 | 0.22 |

Roughly 2015-era Faceness-Net territory — reasonable for a 549k-parameter grayscale CPU model, far from SOTA (~0.97/0.96/0.90) whose advantage is almost entirely in sub-24px faces. Two lessons from the evaluation itself:

1. **Weighted-average NMS drifts.** Merging each cluster by score-weighted averaging (our previous default) smears box positions when thousands of low-score candidates join a cluster, wrecking IoU. Keeping the argmax position (`std` NMS) and sweeping the score threshold afterwards lifted easy-tier AP from 0.007 to 0.559 on identical score maps. Raw candidates are cached (`outputs/cache/wider_cands/`, 3.2GB), so NMS/threshold variants re-evaluate in minutes.
2. **Box shape calibration.** WIDER faces are ~1.26× taller than wide (train-set median h/w) while the detector emits squares. Stretching box height ×1.25 (no vertical shift; matches train stats, so not val overfitting) lifts AP on every tier:

| tier | square | calibrated ×1.25 |
|---|---|---|
| Easy | 0.559 | **0.643** |
| Medium | 0.536 | **0.572** |
| Hard | 0.231 | **0.240** |

The calibration is applied in `detect()` (`BOX_CALIBR` in `deepface/detect.py`).

Remaining failure modes, by face size: <24px faces are physically unreachable (36px minimum window) — that is ~2/3 of all GT and caps Hard AP; >224px faces sit at pyramid depths where the face-to-context ratio leaves the training distribution and the network never fires; the 96–144px band is the sweet spot.

## Failed Experiment: Upscale Pyramid

Zero-training attempt at the tiny-face problem: extend the pyramid *upward* (scales 0.30–0.92, i.e. 1.1×–3.3× enlargement, coarse 1.25 ladder, tiled forward pass to cap RAM) so 8–26px faces enter the detectable band. On a stratified 244-image sample, tiny-face recall genuinely doubled (hard-tier recall@0.8: 0.22 → 0.42), but the upscaled levels flooded the output with false positives (~9× more detections at score ≥ 0.8): wrong-scale duplicates of already-detected faces, plus texture FPs — bilinear-upscaled noise at 3× is apparently face-like enough for the network at the 36px window scale. A cascade filter (drop upscale detections overlapping any base-pyramid detection) plus a separate score threshold for upscale detections never recovered the loss:

| config | easy | medium | hard |
|---|---|---|---|
| base | **0.538** | **0.518** | **0.220** |
| + upscale, cascade, thr 0.2 | 0.505 | 0.474 | 0.211 |
| + upscale, cascade, thr 0.95 | 0.531 | 0.511 | 0.221 |

Conclusion: at these sizes the signal is gone, not merely under-thresholded — direction abandoned. Infrastructure remains in place (`pyramid(min_scale=...)`, `_dense_peaks_tiled`, `tools/eval_wider.py --up/--sample`) if a better model ever justifies it.

## Failed Experiment: Widening the Training Scale Band

Training-side attempt at the same scale problem, zero extra steps. Two variants, each 2000 steps from scratch, evaluated on the same 244-image stratified val sample (std NMS, floor 0.05):

- **A**: widen the positive face-size band from [26,52]px to [20,96]px (`pos_range` in `DeepSceneSampler`) so the dense detector learns more scales directly.
- **B**: alternate 192px and 96px training tiles, mimicking the small inputs of deep pyramid levels.

| model | easy | medium | hard | P@0.85 |
|---|---|---|---|---|
| baseline (narrow band, 192px tiles) | **0.591** | **0.558** | **0.278** | 0.61 |
| A + B | 0.541 | 0.524 | 0.263 | 0.15 |
| A only | 0.258 | 0.288 | 0.143 | 0.07 |

A is the main poison: the narrow band is what forces sharp scale selectivity (fire only at the correct scale); widening it blurs the decision boundary and floods high-score FPs (crop center-acc also degraded 0.98 → 0.87). B partially mitigated A's damage but did not help on its own merits here. Both variants abandoned; default sampler behavior restored (widening kept as the opt-in `pos_range` parameter). The winning configuration remains the baseline recipe — scale coverage is fundamentally a data-complexity/capacity problem, not a labeling trick.

## D Route: Plain Continuation Training (current default)

With both zero-training tricks (upscale pyramid) and labeling tricks (A/B) refuted, the remaining lever was simply more optimization. Continuing training from the baseline weights for another 4000 steps (Adam, lr 3e-4 cosine, same recipe; `tools/train_deep.py 4000 1.0 <out> 0 models/model_deep.pth 192 3e-4`) on the same 244-image sample:

| model | easy | medium | hard | P@0.85 |
|---|---|---|---|---|
| baseline (2000 steps) | 0.591 | 0.558 | 0.278 | 0.61 |
| **D (v1 + 4000 steps)** | **0.638** | **0.602** | **0.301** | **0.80–0.86** |

All three tiers improve, and precision at the default threshold jumps from 0.61 to 0.80+ — continued training mainly suppresses false positives. The continued model is now the default `models/model_deep.pth`; the 2000-step baseline is kept as `models/model_deep_v1.pth`.

## Cross-Level Redundancy: Half the Pyramid Is Free, but Support Doesn't Separate FPs

Two measurements on cached candidates (`tools/level_redundancy.py`, `tools/level_support.py`, 244-image sample):

**Compute saving works.** A face lands in the [26,52]px positive band for ~7 consecutive levels at ladder ratio 1.1, so detections are highly redundant across levels. Dropping every other level (effective ratio 1.21, i.e. half the forward passes) costs almost nothing:

| ladder | easy | medium | hard |
|---|---|---|---|
| ratio 1.1 (all levels) | 0.591 | 0.558 | 0.278 |
| ratio 1.21 (even levels) | 0.569 | 0.550 | 0.274 |
| ratio 1.21 + sub-grid interpolation | **0.602** | **0.571** | **0.285** |

Sub-grid interpolation (`tools/scale_interp.py`, integrated as `scale_refine` in `deepface/detect.py`): fit a parabola to each detection's log-score profile along the ladder and move the box to the fractional peak scale. It not only recovers the thinning loss but beats the full ladder — the 1.1 grid's ±10% size quantization was itself an IoU loss source. The default ladder is now ratio 1.21 (~2x fewer forward passes) with refinement on top.

**The ladder itself can be optimized** (`tools/ladder_opt.py`): choosing level placement to minimize interpolation loss Σ ρ(k)κ(k)d_k² (ρ = matched-face density per level, κ = profile curvature, d = distance to nearest kept level) under an equal compute budget (Σ s_k²), solved exactly by segment DP. Findings: face value concentrates heavily at level 0 (282/680 matched faces peak at full resolution — WIDER's small-face dominance), so the optimizer keeps the expensive top dense (levels 0-1), stretches the middle to gaps of 2-3, and keeps all cheap deep levels. Equal-budget AP gain over the uniform 1.21 ladder is small (+0.005 easy: 0.6087 vs 0.6032), so the uniform ladder stays the default — the optimizer was fit on the val sample — but the tool transfers: run it once per dataset/model to get the optimal ladder for that distribution.

**Cross-level support as an FP filter fails.** The hypothesis was that a real face is corroborated by adjacent-level responses while a texture FP fires alone. In fact 91% of mid-score ([0.3,0.6)) candidates have adjacent-level support and are still ~100% FP: the network's false positives are spatially coherent face-like textures, whose responses are just as scale-smooth as real faces. Support-filter/rescue AP rules gained ≤0.001 or hurt. Direction abandoned; multi-scale consistency is a property of any coherent pattern, not of faces.

**Profile *shape* is real but redundant** (`tools/scale_profile.py`). Treating the response as a continuous scale-space field f(x, y, log s), a true face is an asymmetric bump: fast decay toward smaller scales (mean score ratio at -2 levels: 0.47 for TP vs 0.18 for FP), a plateau tail toward larger scales (ratio ~0.95 at +1 — consistent with the upscale-pyramid duplicates), and a stable peak position across levels (drift 0.44 box-sizes vs 0.61 for FP). The network's learned scale selectivity peaks at the small end of the band (~35-39px face-in-window: it prefers more context). However, adding these profile features to the score does not improve separation (held-out AUC 0.9938 score-only vs 0.9936 with profile; AP within noise): adjacent-level scores are computed from nearly the same pixels, so the profile is the same evidence resampled, not independent evidence.
