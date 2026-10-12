"""Cross-level support as a true/false-positive signal.

Hypothesis: a real face fires on ~7 consecutive pyramid levels (positive band
[26,52]px is 2x wide, ladder ratio 1.1), so a detection with candidates at
adjacent levels (same place, ~1.1x size, score >= 0.3) is likely real; a lone
single-level detection is likely a texture FP. If support separates TP from FP,
we can rescue mid-score real faces (fix threshold-caused misses) while dropping
lone FPs.

Protocol: from cached candidates (floor 0.05), compute pre-NMS adjacent-level
support for each NMS survivor, label TP/FP by official-style greedy IoU-0.5
matching against all valid GT, then report precision by (score band x support)
and AP under filter/rescue rules.

Usage: py -m tools.level_support
"""
import numpy as np
import torch

from deepface.detect import cluster_nms
from deepface.evaluate import load_val_gt, wider_ap
from tools.eval_wider import sample_rels
from tools.level_redundancy import load_cands, iou_matrix

SUPPORT_THR = 0.3   # supporting candidate must clear this score


def support_counts(c, levels, keep_mask):
    """For each candidate in keep_mask, count distinct adjacent-level candidates
    (|dlevel|=1, score >= SUPPORT_THR) with IoU >= 0.3."""
    sup_pool = c[c[:, 4] >= SUPPORT_THR]
    sup_lv = levels[c[:, 4] >= SUPPORT_THR]
    out = np.zeros(len(c), dtype=int)
    for i in np.nonzero(keep_mask)[0]:
        l = levels[i]
        adj = np.abs(sup_lv - l) == 1
        if adj.any():
            out[i] = int((iou_matrix(c[i:i + 1, :4], sup_pool[adj, :4]) >= 0.3).sum())
    return out


def nms_with_support(c, levels, floor=0.05):
    """std NMS at floor; returns (dets[M,5], support[M])."""
    order = c[:, 4].argsort()[::-1]
    c_s, lv_s = c[order], levels[order]
    t = torch.from_numpy(c_s)
    det = cluster_nms(t, floor, 'std').numpy()
    if len(det) == 0:
        return det, np.zeros(0, dtype=int)
    # survivors = highest-score member of each cluster; locate them by exact match
    idx = []
    used = np.zeros(len(c_s), dtype=bool)
    for d in det:
        m = (np.abs(c_s - d).sum(1) < 1e-4) & ~used
        i = int(np.nonzero(m)[0][0])
        used[i] = True
        idx.append(i)
    sup = support_counts(c_s, lv_s, np.isin(np.arange(len(c_s)), idx))
    return det, sup[np.isin(np.arange(len(c_s)), idx)]


def match_gt(det, boxes, valid):
    """Greedy IoU-0.5 matching by descending score. Returns tp flag per det.
    Dets matching invalid (ignored) GT are neither TP nor FP -> returned as -1."""
    if len(det) == 0:
        return np.zeros(0, dtype=int)
    lab = np.zeros(len(det), dtype=int) - 1
    if len(boxes) == 0:
        lab[:] = 1  # everything is FP
        return lab
    gx2 = boxes[:, :2] + boxes[:, 2:]
    ix1 = np.maximum(det[:, None, 0], boxes[None, :, 0])
    iy1 = np.maximum(det[:, None, 1], boxes[None, :, 1])
    ix2 = np.minimum(det[:, None, 2], gx2[None, :, 0])
    iy2 = np.minimum(det[:, None, 3], gx2[None, :, 1])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = ((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))[:, None] \
        + (boxes[:, 2] * boxes[:, 3])[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)
    hit = np.zeros(len(boxes), dtype=bool)
    order = det[:, 4].argsort()[::-1]
    for h in order:
        j = int(iou[h].argmax())
        if iou[h, j] < 0.5:
            lab[h] = 1            # FP
        elif not valid[j]:
            lab[h] = -1           # ignored GT
        elif not hit[j]:
            hit[j] = True
            lab[h] = 0            # TP
        else:
            lab[h] = 1            # duplicate -> FP
    return lab


def main():
    rels, boxes, tier_masks = load_val_gt()
    rels = sample_rels(rels, 4)
    valid = {r: np.zeros(len(boxes[r]), dtype=bool) for r in rels}
    for masks in tier_masks.values():
        for r, m in masks.items():
            if r in valid:
                valid[r] |= m

    all_det, all_sup, all_lab = [], [], []
    dets_cache = {}
    for k, rel in enumerate(rels):
        r = load_cands(rel)
        if r is None:
            continue
        c, levels = r
        det, sup = nms_with_support(c, levels)
        dets_cache[rel] = det
        lab = match_gt(det, boxes.get(rel, np.zeros((0, 4))), valid[rel])
        all_det.append(det)
        all_sup.append(sup)
        all_lab.append(lab)
        if (k + 1) % 50 == 0:
            print(f'{k + 1}/{len(rels)}', flush=True)

    det = np.concatenate(all_det)
    sup = np.concatenate(all_sup)
    lab = np.concatenate(all_lab)

    print('\n=== precision by score band x support (ignored dets excluded) ===')
    bands = [(0.05, 0.3), (0.3, 0.6), (0.6, 0.85), (0.85, 1.01)]
    print(f'{"band":<12}{"sup":<6}{"n":<8}{"TP":<8}{"FP":<8}{"prec":<8}')
    for lo, hi in bands:
        for s_min, s_name in [(0, '0'), (1, '>=1'), (2, '>=2')]:
            m = (det[:, 4] >= lo) & (det[:, 4] < hi) & (sup >= s_min) & (lab >= 0)
            n = m.sum()
            if n == 0:
                continue
            tp = (lab[m] == 0).sum()
            print(f'[{lo:.2f},{hi:.2f})  {s_name:<6}{n:<8}{tp:<8}{n - tp:<8}{tp / n:.3f}')

    print('\n=== AP under support rules ===')
    def ap_of(rule, name):
        out = {}
        off = 0
        for rel in dets_cache:
            n = len(dets_cache[rel])
            out[rel] = rule(dets_cache[rel], sup[off:off + n])
            off += n
        res = wider_ap(out, rels=list(out))
        line = '  '.join(f'{t}={res[t][0]:.4f}' for t in sorted(res))
        print(f'{name:<44s} {line}', flush=True)

    ap_of(lambda d, s: d, 'baseline (all dets)')
    ap_of(lambda d, s: d[(d[:, 4] >= 0.85) | (s >= 1)], 'keep >=0.85, else need sup>=1')
    ap_of(lambda d, s: d[(d[:, 4] >= 0.85) | (s >= 2)], 'keep >=0.85, else need sup>=2')
    ap_of(lambda d, s: d[(d[:, 4] >= 0.6) | (s >= 2)], 'keep >=0.6, else need sup>=2')
    ap_of(lambda d, s: d[s >= 1], 'all dets need sup>=1')


if __name__ == '__main__':
    main()
