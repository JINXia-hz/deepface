import glob
import os

import torch

from deepface.wider import load_annotations

HOLDOUT_DIR = 'data/holdout'
SAMPLES_DIR = 'samples'
WIDER_VAL_IMG = 'data/widerface/WIDER_val/images'
WIDER_VAL_TXT = 'data/widerface/wider_face_split/wider_face_val_bbx_gt.txt'
N_HELDOUT_VAL = 200   # first 200 sorted val images are never trained on


def evaluate(net, tag, loader):
    """Crop-level classification metrics for the heatmap backend (Net)."""
    was_training = net.training
    net.eval()
    correct = 0
    total = 0
    TP = TN = FP = FN = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    print(f"[{tag}] Accuracy: {correct / total:.4f}")
    print(f"[{tag}] TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    if was_training:
        net.train()


def center_accuracy(net, tag, loader):
    """Crop-level accuracy for the deep backend (ResUNet): classify each 36x36
    crop by the sigmoid score at the center pixel."""
    was_training = net.training
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            s = torch.sigmoid(net(x)).squeeze(1)
            correct += ((s[:, 18, 18] > 0.5).long() == y).sum().item()
            total += len(y)
    print(f"[{tag}] center-acc: {correct / max(total, 1):.4f}")
    if was_training:
        net.train()


def eval_jobs():
    """(group, path) list: 100 face-free holdout images, 4 samples, and the
    200 held-out WIDER val images."""
    jobs = [('holdout', p) for p in sorted(glob.glob(os.path.join(HOLDOUT_DIR, '*.jpg')))]
    jobs += [('samples', p) for p in sorted(glob.glob(os.path.join(SAMPLES_DIR, '*.jpg')))]
    ann = load_annotations(WIDER_VAL_TXT)
    jobs += [('wider', os.path.join(WIDER_VAL_IMG, rel))
             for rel in sorted(ann)[:N_HELDOUT_VAL]]
    return jobs


def wider_recall(det_fn, min_side=28, iou_thr=0.3):
    """Recall on the 200 held-out WIDER val images (faces with side >= min_side).
    det_fn(abs_image_path) -> [M,5] boxes. Returns (hits, total)."""
    ann = load_annotations(WIDER_VAL_TXT)
    hit = tot = 0
    for rel in sorted(ann)[:N_HELDOUT_VAL]:
        det = det_fn(os.path.join(WIDER_VAL_IMG, rel))
        for x, y, w, h in ann[rel][:, :4].astype(float).tolist():
            if min(w, h) < min_side:
                continue
            tot += 1
            if len(det) == 0:
                continue
            ix1 = det[:, 0].clamp(min=x)
            iy1 = det[:, 1].clamp(min=y)
            ix2 = det[:, 2].clamp(max=x + w)
            iy2 = det[:, 3].clamp(max=y + h)
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            union = ((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) + w * h - inter)
            hit += (inter / union).max().item() >= iou_thr
    return hit, tot
