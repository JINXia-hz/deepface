# deepface — Fully Convolutional Heatmap Face Detection

Detect all faces in any image using a CNN binary classifier (36×36 grayscale input, face/non-face output).

## Directory Structure

```
├── net.py              # CNN model definition (Net: 3 conv layers + FC, input 1×36×36)
├── load_data.py        # Dataset loading (preloaded into memory, returns train/valid/test DataLoaders)
├── train.py            # Training; usage: py train.py <epochs>; model saved to models/
├── test.py             # Evaluate the model on the test set (Accuracy / TP / TN / FP / FN)
├── detect_heatmap.py   # Fully convolutional heatmap face detection (current approach)
├── export_mistakes.py  # Export a grid of misclassified test samples to outputs/
├── models/             # Trained model weights
│   ├── model.pth       # Legacy model (trained with per-window histogram equalization, test 98.77%)
│   └── model_noeq.pth  # Current model (no equalization, test 97.93%, used with detect_heatmap.py)
├── train_images/       # Training set (0=non-face, 1=face, 36×36)
├── test_images/        # Test set
├── samples/            # Sample images to detect on
└── outputs/            # Detection outputs (detect_output.png, etc.)
```

## Usage

```bash
py train.py 10                        # Train for 10 epochs, saves models/model_noeq.pth
py test.py                            # Evaluate on the test set
py detect_heatmap.py samples/test.jpg # Detect; draws boxes and saves outputs/detect_output.png
py export_mistakes.py                 # Export misclassified samples
```

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
