import sys

import numpy as np

from deepface.evaluate import load_val_gt, wider_ap
from tools.eval_wider import load_dets


def transform(dets, a, b, c):
    """Square box (side L, center cx,cy) -> width a*L, height b*L,
    center shifted down by c*L."""
    out = {}
    for rel, d in dets.items():
        if len(d) == 0:
            out[rel] = d
            continue
        d = d.copy()
        L = d[:, 2] - d[:, 0]
        cx = (d[:, 0] + d[:, 2]) / 2
        cy = (d[:, 1] + d[:, 3]) / 2 + c * L
        d[:, 0] = cx - a * L / 2
        d[:, 2] = cx + a * L / 2
        d[:, 1] = cy - b * L / 2
        d[:, 3] = cy + b * L / 2
        out[rel] = d
    return out


def main(variants):
    rels, _, _ = load_val_gt()
    dets = load_dets('deep', rels)
    print(f"{len(dets)} images, baseline first")
    print(f"{'a(w)':<6}{'b(h)':<6}{'c(dy)':<7}{'easy':<8}{'medium':<8}{'hard':<8}", flush=True)
    for a, b, c in variants:
        res = wider_ap(transform(dets, a, b, c))
        ap = {t: v[0] for t, v in res.items()}
        print(f"{a:<6}{b:<6}{c:<7}{ap['easy']:<8.4f}{ap['medium']:<8.4f}{ap['hard']:<8.4f}",
              flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        variants = [tuple(float(v) for v in s.split(',')) for s in sys.argv[1:]]
    else:
        variants = [(a, b, c)
                    for a in (0.9, 1.0)
                    for b in (1.1, 1.2, 1.3, 1.4)
                    for c in (-0.1, 0.0, 0.1)]
        variants = [(1.0, 1.0, 0.0)] + variants
    main(variants)
