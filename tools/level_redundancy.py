"""Measure cross-pyramid-level redundancy from cached candidates.

Questions:
1. Level occupancy: how many levels contain zero candidates >= 0.3 (pure waste)?
2. Adjacent support: do strong detections (>= 0.85) have a >= 0.3 candidate at an
   adjacent level (same place, ~1.1x size)? I.e. is a face visible across levels?
3. Level-thinning AP: drop every other level (simulate ratio 1.21) and re-evaluate
   AP on the 244-image sample vs all levels.

Usage: py -m tools.level_redundancy [n_images_for_12]
"""
import os
import sys
import numpy as np
import torch

from deepface.detect import cluster_nms
from deepface.evaluate import load_val_gt, wider_ap
from tools.eval_wider import sample_rels, cand_dir


def load_cands(rel):
    fn = rel.replace('/', '__') + '.pt'
    p = os.path.join(cand_dir('deep', False, ''), fn)
    if not os.path.exists(p):
        return None
    c = torch.load(p, weights_only=True).numpy()
    sizes = c[:, 2] - c[:, 0]
    levels = np.round(np.log(np.maximum(sizes, 1) / 36) / np.log(1.1)).astype(int)
    return c, levels


def iou_matrix(a, b):
    """a: [N,4], b: [M,4] -> [N,M] IoU."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def occupancy_and_support(rels):
    occ_levels, occ_total = [], 0
    sup_hits, sup_tot = 0, 0
    for rel in rels:
        r = load_cands(rel)
        if r is None:
            continue
        c, levels = r
        lv = np.unique(levels)
        span = np.arange(lv.min(), lv.max() + 1)
        occ_total += len(span)
        occ_levels.append([np.any((levels == l) & (c[:, 4] >= 0.3)) for l in span])
        strong = c[c[:, 4] >= 0.85]
        sl = levels[c[:, 4] >= 0.85]
        for box, l in zip(strong, sl):
            adj = (np.abs(levels - l) == 1) & (c[:, 4] >= 0.3)
            sup_tot += 1
            if adj.any() and (iou_matrix(box[None, :4], c[adj, :4]) >= 0.3).any():
                sup_hits += 1
    occ = np.concatenate(occ_levels)
    print(f'levels scanned: {occ_total}, empty at 0.3: {(~occ).mean():.3f}')
    print(f'adjacent-level support for >=0.85 detections: {sup_hits}/{sup_tot} '
          f'= {sup_hits / max(sup_tot, 1):.3f}')


def thinned_ap(rels):
    def build(filter_odd):
        dets = {}
        for rel in rels:
            r = load_cands(rel)
            if r is None:
                continue
            c, levels = r
            if filter_odd:
                keep = levels % 2 == 0
                c = c[keep]
            c = torch.from_numpy(c)
            if len(c):
                c = c[c[:, 4].argsort(descending=True)]
            dets[rel] = cluster_nms(c, 0.05, 'std').numpy()
        return dets

    for name, odd in [('all levels (ratio 1.1)', False), ('even levels only (ratio 1.21)', True)]:
        dets = build(odd)
        res = wider_ap(dets, rels=rels)
        line = '  '.join(f'{t}={res[t][0]:.4f}' for t in sorted(res))
        print(f'{name:34s} {line}')


if __name__ == '__main__':
    rels, _, _ = load_val_gt()
    rels = sample_rels(rels, 4)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    occupancy_and_support(rels[:n])
    thinned_ap(rels)
