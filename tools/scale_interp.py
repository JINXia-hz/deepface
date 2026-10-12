"""Sub-grid scale interpolation on a thinned pyramid.

A face's score profile along the log-scale axis is a bump peaking where the
face appears ~35-39px in the window. On a thinned ladder (every other level,
ratio 1.21) the true peak usually falls between sampled levels; fitting a
parabola to log-score vs level index through the survivor and its two coarse
neighbors recovers a fractional peak level k* and peak score s*, giving
sub-grid box size (36 * 1.1**k*) and a corrected score.

Simulated honestly from cached ratio-1.1 candidates: detection and fitting
use even levels only. Compares AP (244-image sample):
  baseline all levels / even only / even + size interp / even + size+score interp

Usage: py -m tools.scale_interp
"""
import numpy as np
import torch

from deepface.detect import cluster_nms
from deepface.evaluate import load_val_gt, wider_ap
from tools.eval_wider import sample_rels
from tools.level_redundancy import load_cands, iou_matrix


def interp_dets(c, levels, mode):
    """Even-level detection + parabola interpolation. mode: 'none' | 'size' | 'both'."""
    even = levels % 2 == 0
    ce, le = c[even], levels[even]
    if len(ce) == 0:
        return ce
    t = torch.from_numpy(ce)
    det = cluster_nms(t[t[:, 4].argsort(descending=True)], 0.05, 'std').numpy()
    if mode == 'none' or len(det) == 0:
        return det
    out = det.copy()
    for i, d in enumerate(det):
        size = d[2] - d[0]
        k = int(round(np.log(size / 36) / np.log(1.1)))
        pts = [(0.0, d[4])]
        for dk in (-2, 2):
            m = le == k + dk
            if not m.any():
                pts = None
                break
            ious = iou_matrix(d[None, :4], ce[m, :4])[0]
            j = int(ious.argmax())
            if ious[j] < 0.2:
                pts = None
                break
            pts.append((float(dk), ce[m][j, 4]))
        if pts is None or len(pts) < 3:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.log(np.maximum([p[1] for p in pts], 1e-6))
        a, b, cc = np.polyfit(xs, ys, 2)
        if a >= 0:
            continue
        x_star = -b / (2 * a)
        if abs(x_star) > 2:
            continue
        s_star = float(np.exp(a * x_star ** 2 + b * x_star + cc))
        cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
        new_size = 36 * 1.1 ** (k + x_star)
        out[i, 0], out[i, 1] = cx - new_size / 2, cy - new_size / 2
        out[i, 2], out[i, 3] = cx + new_size / 2, cy + new_size / 2
        if mode == 'both':
            out[i, 4] = max(d[4], s_star)
    return out


def main():
    rels, _, _ = load_val_gt()
    rels = sample_rels(rels, 4)

    dets_all, dets_even, dets_size, dets_both = {}, {}, {}, {}
    for k, rel in enumerate(rels):
        r = load_cands(rel)
        if r is None:
            continue
        c, levels = r
        t = torch.from_numpy(c)
        dets_all[rel] = cluster_nms(t[t[:, 4].argsort(descending=True)], 0.05, 'std').numpy()
        dets_even[rel] = interp_dets(c, levels, 'none')
        dets_size[rel] = interp_dets(c, levels, 'size')
        dets_both[rel] = interp_dets(c, levels, 'both')
        if (k + 1) % 50 == 0:
            print(f'{k + 1}/{len(rels)}', flush=True)

    for name, d in [('all levels (ratio 1.1)', dets_all),
                    ('even only (ratio 1.21)', dets_even),
                    ('even + size interp', dets_size),
                    ('even + size+score interp', dets_both)]:
        res = wider_ap(d, rels=list(d))
        print(f'{name:30s}', '  '.join(f'{t}={res[t][0]:.4f}' for t in sorted(res)), flush=True)


if __name__ == '__main__':
    main()
