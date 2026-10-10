import sys

import torch

from deepface.detect import build_G, cluster_nms
from deepface.evaluate import wider_recall

DEFAULT_THR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 0.995, 0.999]
DEFAULT_METHODS = ['wavg']
KERNEL_PATH = 'outputs/cache/kernel.pt'  # only needed by the 'softk' method


def main(cache_paths, thresholds, methods):
    G = None
    if 'softk' in methods:
        G = build_G(torch.load(KERNEL_PATH, weights_only=True))

    caches = {p: torch.load(p, weights_only=False) for p in cache_paths}
    # sort each image's candidates by score once
    for data in caches.values():
        for k, v in data.items():
            if len(v):
                data[k] = v[v[:, 4].argsort(descending=True)]

    for path, data in caches.items():
        print(f"\n=== {path} ===")
        print(f"{'method':<10}{'thr':<8}{'holdoutFP':<12}{'widerRecall':<13}samples")
        for method in methods:
            for thr in thresholds:
                def det_fn(img_path, m=method, t=thr):
                    return cluster_nms(data.get(('wider', img_path), torch.zeros(0, 5)),
                                       t, m, G=G)

                fp = sum(len(cluster_nms(data.get(k, torch.zeros(0, 5)), thr, method, G=G))
                         for k in data if k[0] == 'holdout')
                hit, tot = wider_recall(det_fn)
                smp = [len(cluster_nms(data.get(k, torch.zeros(0, 5)), thr, method, G=G))
                       for k in sorted(data) if k[0] == 'samples']
                print(f"{method:<10}{thr:<8}{fp:<12}{hit / max(tot, 1):<13.4f}{smp}", flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    methods = DEFAULT_METHODS
    thresholds = DEFAULT_THR
    if '--methods' in args:
        i = args.index('--methods')
        methods = args.pop(i + 1).split(',')
        args.pop(i)
    if '--thr' in args:
        i = args.index('--thr')
        thresholds = [float(v) for v in args.pop(i + 1).split(',')]
        args.pop(i)
    main(args or ['outputs/cache/cands_deep.pt'], thresholds, methods)
