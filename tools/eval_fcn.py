import os
import sys

import torch
from PIL import Image

from deepface.data import get_data_loaders
from deepface.detect import detect
from deepface.evaluate import evaluate, center_accuracy, wider_recall, eval_jobs
from deepface.models import load_model

torch.set_num_threads(os.cpu_count() or 1)


def main(model_path):
    net, backend = load_model(model_path)

    # 1) crop-level test accuracy (recall anchor)
    _, _, test_loader = get_data_loaders(batch_size=128)
    if backend == 'heatmap':
        evaluate(net, 'test crops', test_loader)
    else:
        center_accuracy(net, 'test crops', test_loader)

    jobs = eval_jobs()
    det_live = lambda path: detect(net, Image.open(path), backend)

    # 2) false-positive boxes on 100 person-free holdout images
    total = dirty = 0
    for group, path in jobs:
        if group != 'holdout':
            continue
        n = len(det_live(path))
        total += n
        dirty += n > 0
    print(f"holdout: {total} FP boxes across 100 images ({dirty} images with >=1)")

    # 3) detection counts on the four sample images
    for group, path in jobs:
        if group == 'samples':
            print(f"sample {os.path.basename(path)}: {len(det_live(path))} faces")

    # 4) recall on the 200 held-out WIDER val images (faces with side >= 28)
    hit, tot = wider_recall(det_live)
    print(f"wider holdout recall@IoU0.3 (side>=28): {hit}/{tot} = {hit / max(tot, 1):.4f}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'models/model_noeq.pth')
