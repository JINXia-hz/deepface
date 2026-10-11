import os
import sys
import time

import numpy as np
import torch

DET_DIR = 'outputs/cache/wider_dets'
CAND_DIR = 'outputs/cache/wider_cands'
WIDER_VAL_IMG = 'data/widerface/WIDER_val/images'
UP_MIN_SCALE = 0.30   # upscale ladder 0.30 -> 0.92 (up to ~3.3x, for 8-26px faces)
UP_RATIO = 1.25       # coarser ladder than the base pyramid: 6 levels, ~2.6x cheaper
UP_MAX_CANDS = 200_000   # cap per-image upscale candidates (score-sorted)

_net = None
_backend = 'deep'
_up = False


def _init(model_path, backend, up):
    global _net, _backend, _up
    torch.set_num_threads(1)
    from deepface.models import load_model
    _net, _ = load_model(model_path)
    _backend, _up = backend, up


def cand_dir(backend, up):
    return os.path.join(CAND_DIR, backend + ('_up' if up else ''))


def _scan_one(rel):
    from PIL import Image
    from deepface.detect import BACKENDS, SCANNERS
    cfg = BACKENDS[_backend]
    img = Image.open(os.path.join(WIDER_VAL_IMG, rel))
    if _up:
        # upscale levels only; base levels are already cached
        cands = SCANNERS[_backend](_net, img, cfg['floor'], UP_RATIO,
                                   min_scale=UP_MIN_SCALE, max_scale=0.99)
    else:
        cands = SCANNERS[_backend](_net, img, cfg['floor'], cfg['ratio'])
    if len(cands):
        cands = cands[cands[:, 4].argsort(descending=True)][:UP_MAX_CANDS]
    torch.save(cands.float(),
               os.path.join(cand_dir(_backend, _up), rel.replace('/', '__') + '.pt'))
    return rel, len(cands)


def sample_rels(rels, k):
    """Deterministic stratified sample: every ~len/k-th image per event."""
    from itertools import groupby
    out = []
    for _, group in groupby(rels, key=lambda r: r.split('/')[0]):
        g = list(group)
        out += g[::max(1, len(g) // k)][:k]
    return out


def scan(model_path, backend, limit, workers, sample, up):
    from concurrent.futures import ProcessPoolExecutor
    from deepface.evaluate import load_val_gt
    rels, _, _ = load_val_gt()
    if sample:
        rels = sample_rels(rels, sample)
    if limit:
        rels = rels[:limit]
    out_dir = cand_dir(backend, up)
    os.makedirs(out_dir, exist_ok=True)
    todo = [r for r in rels
            if not os.path.exists(os.path.join(out_dir, r.replace('/', '__') + '.pt'))]
    print(f"{len(rels)} images, {len(todo)} to scan "
          f"(backend={backend}, up={up}, workers={workers})")
    t0 = time.time()
    done = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init, initargs=(model_path, backend, up)) as ex:
            for rel, n in ex.map(_scan_one, todo, chunksize=4):
                done += 1
                if done % 20 == 0:
                    rate = done / (time.time() - t0)
                    print(f"{done}/{len(todo)}  {rate:.2f} img/s  "
                          f"eta {(len(todo) - done) / rate / 60:.0f} min", flush=True)
    else:
        _init(model_path, backend, up)
        for rel in todo:
            _scan_one(rel)
            done += 1
            if done % 10 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(todo)}  {rate:.2f} img/s  "
                      f"eta {(len(todo) - done) / rate / 60:.0f} min", flush=True)
    print(f"scan done: {done} images in {(time.time() - t0) / 60:.1f} min")


def load_dets(backend, rels, nms='std', floor=None, up=False):
    """Final detector outputs: cached per image; built from raw candidates with
    cluster_nms at the candidate floor on first use. up=True merges the upscale
    pyramid candidates."""
    from deepface.detect import BACKENDS, cluster_nms
    floor = BACKENDS[backend]['floor'] if floor is None else floor
    det_dir = os.path.join(DET_DIR, f'{backend}_{nms}{floor}' + ('_up' if up else ''))
    os.makedirs(det_dir, exist_ok=True)
    dets = {}
    for rel in rels:
        fn = rel.replace('/', '__') + '.pt'
        det_path = os.path.join(det_dir, fn)
        if os.path.exists(det_path):
            dets[rel] = torch.load(det_path, weights_only=True).numpy()
            continue
        cand_path = os.path.join(cand_dir(backend, False), fn)
        if not os.path.exists(cand_path):
            continue
        cands = torch.load(cand_path, weights_only=True)
        if up:
            up_path = os.path.join(cand_dir(backend, True), fn)
            if os.path.exists(up_path):
                cands = torch.cat([cands, torch.load(up_path, weights_only=True)])
        if len(cands):
            cands = cands[cands[:, 4].argsort(descending=True)]
        det = cluster_nms(cands, floor, nms)
        torch.save(det.float(), det_path)
        dets[rel] = det.numpy()
    return dets


def report(backend, subset, nms, floor, up, sample):
    from deepface.detect import BACKENDS
    from deepface.evaluate import load_val_gt, wider_ap
    rels, boxes, tier_masks = load_val_gt()
    if sample:
        rels = sample_rels(rels, sample)
    dets = load_dets(backend, rels, nms, floor, up)
    print(f"loaded detections for {len(dets)}/{len(rels)} images "
          f"(nms={nms}, floor={BACKENDS[backend]['floor'] if floor is None else floor}, up={up})")
    if subset:
        rels = sorted(dets)
    results = wider_ap(dets, rels=rels)
    grid = 1 - np.arange(1, 1001) / 1000
    print(f"{'tier':<8}{'AP':<8}{f'P@{BACKENDS[backend]['thr']}':<10}{'R':<10}")
    for tier, (ap, prec, rec) in sorted(results.items()):
        # operating point: grid row nearest the backend's default (normalized) thr
        all_s = np.concatenate([d[:, 4] for d in dets.values() if len(d)])
        thr_n = (BACKENDS[backend]['thr'] - all_s.min()) / max(all_s.max() - all_s.min(), 1e-9)
        i = int(np.abs(grid - thr_n).argmin())
        print(f"{tier:<8}{ap:<8.4f}{prec[i]:<10.4f}{rec[i]:<10.4f}")
    np.savez(os.path.join(DET_DIR, f'pr_{backend}_{nms}{"_up" if up else ""}.npz'),
             **{tier: np.stack([p, r]) for tier, (_, p, r) in results.items()})


if __name__ == '__main__':
    args = sys.argv[1:]
    cmd = args.pop(0) if args else 'ap'

    def _opt(name, default):
        if name in args:
            i = args.index(name)
            v = args.pop(i + 1)
            args.pop(i)
            return v
        return default

    backend = _opt('--backend', 'deep')
    up = '--up' in args
    sample = int(_opt('--sample', 0))
    if cmd == 'scan':
        scan(_opt('--model', 'models/model_deep.pth'), backend,
             int(_opt('--limit', 0)), int(_opt('--workers', '7')), sample, up)
    elif cmd == 'ap':
        floor = _opt('--floor', None)
        report(backend, subset='--subset' in args,
               nms=_opt('--nms', 'std'), floor=float(floor) if floor else None,
               up=up, sample=sample)
