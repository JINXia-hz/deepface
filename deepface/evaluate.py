import glob
import os

import numpy as np
import torch

from deepface.wider import load_annotations

HOLDOUT_DIR = 'data/holdout'
SAMPLES_DIR = 'samples'
WIDER_VAL_IMG = 'data/widerface/WIDER_val/images'
WIDER_VAL_TXT = 'data/widerface/wider_face_split/wider_face_val_bbx_gt.txt'
GT_MAT_DIR = 'data/widerface/eval_tools/ground_truth'
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


def _cellstr(x):
    while isinstance(x, np.ndarray):
        x = x.flat[0]
    return str(x)


def load_val_gt():
    """Official WIDER val ground truth from the eval_tools mats.
    Returns (rels, boxes, tier_masks):
      rels: 'event/name.jpg' for all 3226 val images, official order
      boxes: rel -> float32 [N,4] (x,y,w,h), every annotated box
      tier_masks: 'easy'/'medium'/'hard' -> rel -> bool [N] tier membership"""
    import scipy.io as sio
    rels, boxes, tier_masks = [], {}, {}
    for tier in ('face', 'easy', 'medium', 'hard'):
        m = sio.loadmat(os.path.join(GT_MAT_DIR, f'wider_{tier}_val.mat'))
        for e in range(61):
            ev = _cellstr(m['event_list'][e])
            files = m['file_list'][e][0]
            bbxs = m['face_bbx_list'][e][0]
            gts = m['gt_list'][e][0] if tier != 'face' else None
            for j in range(files.shape[0]):
                rel = f"{ev}/{_cellstr(files[j])}.jpg"
                if tier == 'face':
                    rels.append(rel)
                    boxes[rel] = np.asarray(bbxs[j][0], dtype=np.float32).reshape(-1, 4)
                else:
                    n = len(boxes[rel])
                    mask = np.zeros(n, dtype=bool)
                    idx = np.asarray(gts[j][0]).reshape(-1).astype(int) - 1
                    mask[idx] = True
                    tier_masks.setdefault(tier, {})[rel] = mask
    return rels, boxes, tier_masks


def voc_ap(rec, prec):
    """VOC all-point interpolated AP (port of eval_tools/plot/VOCap.m)."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def pr_curve(dets, boxes, tier_mask, thresholds):
    """Official WIDER matching at IoU 0.5 (port of eval_tools/evaluation.m).

    dets: rel -> float32 [M,5] (x1,y1,x2,y2,score). Detections matching a
    non-tier box are ignored (neither TP nor FP); tier boxes matched once.
    Scores are globally min-max normalized like eval_tools/norm_score.m.
    Returns (prec, rec) sampled at the given descending score thresholds.
    """
    all_s = np.concatenate([d[:, 4] for d in dets.values() if len(d)]) \
        if any(len(d) for d in dets.values()) else np.array([0.0, 1.0])
    smin, smax = all_s.min(), all_s.max()
    span = max(smax - smin, 1e-9)

    pr = np.zeros((len(thresholds), 2))
    count_face = int(sum(m.sum() for m in tier_mask.values()))
    for rel, b in boxes.items():
        if len(b) == 0:
            continue
        det = dets.get(rel)
        if det is None or len(det) == 0:
            continue
        det = np.asarray(det, dtype=np.float32)
        det = det[det[:, 4].argsort()[::-1]]
        s = (det[:, 4] - smin) / span
        tier = tier_mask[rel]

        gx2 = b[:, :2] + b[:, 2:]
        ix1 = np.maximum(det[:, None, 0], b[None, :, 0])
        iy1 = np.maximum(det[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(det[:, None, 2], gx2[None, :, 0])
        iy2 = np.minimum(det[:, None, 3], gx2[None, :, 1])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = ((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))[:, None] \
            + (b[:, 2] * b[:, 3])[None, :] - inter
        iou = inter / np.maximum(union, 1e-9)   # [M,N]

        tier_hit = np.zeros(len(b), dtype=bool)
        proposal = np.ones(len(det), dtype=bool)
        rec_h = np.zeros(len(det))
        for h in range(len(det)):
            j = int(iou[h].argmax())
            if iou[h, j] >= 0.5:
                if not tier[j]:
                    proposal[h] = False
                elif not tier_hit[j]:
                    tier_hit[j] = True
            rec_h[h] = tier_hit.sum()

        cprop = np.cumsum(proposal)
        k = np.searchsorted(-s, -np.asarray(thresholds), side='right')
        valid = k > 0
        pr[valid, 0] += cprop[k[valid] - 1]
        pr[valid, 1] += rec_h[k[valid] - 1]

    prec = pr[:, 1] / np.maximum(pr[:, 0], 1)
    rec = pr[:, 1] / max(count_face, 1)
    return prec, rec


def wider_ap(dets, rels=None, thresh_num=1000):
    """Official WIDER val AP per difficulty tier.
    dets: rel -> [M,5]. rels: restrict to this subset (default: all 3226).
    Returns {tier: (ap, prec_curve, rec_curve)}."""
    all_rels, boxes, tier_masks = load_val_gt()
    keep = set(rels) if rels is not None else set(all_rels)
    boxes = {r: b for r, b in boxes.items() if r in keep}
    grid = 1 - np.arange(1, thresh_num + 1) / thresh_num
    out = {}
    for tier, masks in tier_masks.items():
        prec, rec = pr_curve(dets, boxes, {r: m for r, m in masks.items() if r in keep}, grid)
        out[tier] = (voc_ap(rec, prec), prec, rec)
    return out
