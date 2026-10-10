# deepface — Fully Convolutional Heatmap Face Detection

Detect all faces in any image. The default detector is a ResUNet dense score-map model; the original CNN binary classifier (36×36 grayscale input, face/non-face output) with its fully-convolutional heatmap pipeline is kept for comparison.

All shared logic lives in the `deepface/` package; runnable scripts live in `tools/` and are invoked as modules from the repo root (`py -m tools.<name> ...`), so no path hacks are needed anywhere.

## Directory Structure

```
├── deepface/           # Package: all shared logic (no side effects on import)
│   ├── models.py       # Net (36×36 window classifier) + ResUNet (dense score map) + load_model()
│   ├── data.py         # Crop dataset loading (preloaded into memory, train/valid/test DataLoaders)
│   ├── wider.py        # WIDER FACE annotation parsing, scene tile samplers, stride-8/stride-1 labels
│   ├── detect.py       # Pyramid scanning for both backends, weighted/cluster NMS, detect()
│   └── evaluate.py     # Crop metrics, eval job lists, WIDER held-out recall
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
│   └── measure_kernel.py   # Importance kernel / response envelope / GT face spacing
├── models/             # Trained model weights
│   ├── model.pth       # Legacy model (trained with per-window histogram equalization, test 98.77%)
│   ├── model_noeq.pth  # Baseline (no equalization, test 97.93%)
│   ├── model_fcn.pth   # FCN fine-tuned on WIDER val scenes
│   ├── model_fcn2.pth  # FCN fine-tuned on WIDER train+val scenes, 3000 steps
│   └── model_deep.pth  # ResUNet trained from scratch on WIDER scenes (default detector)
├── data/               # Scene data (git-ignored): COCO val2017, no_person, holdout, widerface
├── train_images/       # Training set (0=non-face, 1=face, 36×36)
├── test_images/        # Test set
├── samples/            # Sample images to detect on
└── outputs/            # Detection outputs (<name>_<n>faces.png), cache/ for candidate caches
```

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
