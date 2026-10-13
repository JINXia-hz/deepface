"""Diagnose missed faces: probe net score vs in-plane rotation and scale.

For each target (image, face center, approx face side), crop a square of
side * 60/36 around the center (the net's 60px receptive field for a 36px
window), rotate it through a range of angles, resize to 60x60 with the same
PIL BILINEAR used in training/detection, and read the center-pixel score.

If score peaks high at some off-axis angle, the miss is rotation sensitivity;
if it stays low at every angle, the cause is occlusion/domain, not pose.

A detected face probed the same way serves as control (should peak near 0).

    py -m tools.diag_miss
"""

import numpy as np
import torch
from PIL import Image

from deepface.models import load_model

MODEL = 'models/model_deep.pth'
FIELD = 60 / 36   # receptive field per window side

# (image, cx, cy, face side, note)
TARGETS = [
    ('samples/20200311181548-5e692ad4c84d3.jpg', 1285, 500, 150, 'veiled woman (missed)'),
    ('samples/20200311181548-5e692ad4c84d3.jpg', 460, 1475, 120, 'child (detected, control)'),
    ('samples/20200410000148-5e8f46ecb0106.jpg', 3665, 2500, 250, 'lying man, ~90deg (missed)'),
    ('samples/20200410000148-5e8f46ecb0106.jpg', 3480, 1875, 110, 'seated woman (detected, control)'),
]
ANGLES = [-135, -112, -90, -67, -45, -30, -15, 0, 15, 30, 45, 67, 90, 112, 135, 180]


def center_score(net, patch):
    """patch: 60x60 grayscale PIL -> sigmoid score at the center pixel."""
    t = torch.from_numpy(np.array(patch)).float().mul_(2.0 / 255.0).sub_(1.0)
    with torch.no_grad():
        prob = torch.sigmoid(net(t[None, None]))[0, 0]
    return prob[prob.shape[0] // 2, prob.shape[1] // 2].item()


def probe(net, path, cx, cy, side):
    img = Image.open(path).convert('L')
    rows = []
    for sf in (0.85, 1.0, 1.2):   # face-side uncertainty
        L = int(side * sf * FIELD)
        crop = img.crop((cx - L // 2, cy - L // 2, cx + L // 2, cy + L // 2))
        scores = []
        for a in ANGLES:
            rot = crop.rotate(-a, resample=Image.BILINEAR, fillcolor=128)
            scores.append(center_score(net, rot.resize((60, 60), Image.BILINEAR)))
        rows.append((sf, scores))
    return rows


def main():
    net, backend = load_model(MODEL)
    print(f'model={MODEL} backend={backend}')
    for path, cx, cy, side, note in TARGETS:
        rows = probe(net, path, cx, cy, side)
        print(f'\n== {note}  ({path.split("/")[-1]}, center=({cx},{cy}), side~{side})')
        print('angle:  ' + '  '.join(f'{a:>5d}' for a in ANGLES))
        for sf, scores in rows:
            print(f'side*{sf:.2f}: ' + '  '.join(f'{s:5.2f}' for s in scores)
                  + f'   max={max(scores):.3f} @{ANGLES[int(np.argmax(scores))]}deg')


if __name__ == '__main__':
    main()
