"""Scale-space profile shape as a TP/FP signature.

The binary "has adjacent support" test failed (textures also fire across
levels). This tests a richer hypothesis: the *shape* of the response profile
along the log-scale axis differs between real faces (bump shaped by the
[26,52] training band: rise, peak ~39px, fall; stable center) and face-like
textures (flat/monotone/wandering).

Per NMS survivor: scores at levels +-1, +-2 (best-IoU match, IoU>=0.2) as
ratios to the survivor score, center drift relative to box size, and profile
curvature. Label TP/FP by official-style IoU-0.5 greedy matching. Report
feature stats and held-out AUC: score alone vs score + profile features
(logistic regression trained on half the images). If separable, also report
AP with learned rescoring on the held-out half.

Usage: py -m tools.scale_profile
"""
import numpy as np
import torch

from deepface.evaluate import load_val_gt, wider_ap
from tools.eval_wider import sample_rels
from tools.level_redundancy import load_cands, iou_matrix
from tools.level_support import nms_with_support, match_gt

DELTAS = (-2, -1, 1, 2)


def profile_features(c, levels, det):
    """det: [M,5] NMS survivors (subset of c). Returns [M, F] features."""
    # locate survivors inside c by exact value match
    order = c[:, 4].argsort()[::-1]
    c_s, lv_s = c[order], levels[order]
    idx = []
    used = np.zeros(len(c_s), dtype=bool)
    for d in det:
        m = (np.abs(c_s - d).sum(1) < 1e-4) & ~used
        i = int(np.nonzero(m)[0][0])
        used[i] = True
        idx.append(i)

    feats = np.zeros((len(det), 2 * len(DELTAS) + 2))
    for row, i in enumerate(idx):
        s0 = c_s[i, 4]
        l0 = lv_s[i]
        box = c_s[i, :4]
        size = max(box[2] - box[0], 1e-6)
        col = 0
        ratio = {}
        for d in DELTAS:
            m = lv_s == l0 + d
            r, drift = 0.0, 1.0
            if m.any():
                ious = iou_matrix(box[None, :], c_s[m, :4])[0]
                j = int(ious.argmax())
                if ious[j] >= 0.2:
                    pool = c_s[m]
                    b2 = pool[j]
                    r = float(b2[4] / max(s0, 1e-6))
                    drift = float(np.hypot((b2[0] + b2[2] - box[0] - box[2]) / 2,
                                           (b2[1] + b2[3] - box[1] - box[3]) / 2) / size)
            feats[row, col] = r
            feats[row, col + 1] = drift
            ratio[d] = r
            col += 2
        # curvature of the log-scale profile (bump sharpness), relative to s0
        feats[row, col] = (ratio[-1] + ratio[1]) / 2 - 1.0
        feats[row, col + 1] = s0
    return feats


def auc(scores, labels):
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def logistic_fit(X, y, epochs=400, lr=0.5):
    Xmu, Xsd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - Xmu) / Xsd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt)
        loss.backward()
        opt.step()
    return w.detach(), b.detach(), Xmu, Xsd


def main():
    rels, boxes, tier_masks = load_val_gt()
    rels = sample_rels(rels, 4)
    valid = {r: np.zeros(len(boxes[r]), dtype=bool) for r in rels}
    for masks in tier_masks.values():
        for r, m in masks.items():
            if r in valid:
                valid[r] |= m

    per_rel = {}
    for k, rel in enumerate(rels):
        r = load_cands(rel)
        if r is None:
            continue
        c, levels = r
        det, _ = nms_with_support(c, levels)
        if len(det) == 0:
            per_rel[rel] = (det, np.zeros((0, 12)), np.zeros(0, dtype=int))
            continue
        feats = profile_features(c, levels, det)
        lab = match_gt(det, boxes.get(rel, np.zeros((0, 4))), valid[rel])
        per_rel[rel] = (det, feats, lab)
        if (k + 1) % 50 == 0:
            print(f'{k + 1}/{len(rels)}', flush=True)

    names = ['r-2', 'drift-2', 'r-1', 'drift-1', 'r+1', 'drift+1', 'r+2', 'drift+2',
             'curv', 'score']
    X = np.concatenate([f for _, f, _ in per_rel.values()])
    y = np.concatenate([l for _, _, l in per_rel.values()])
    keep = y >= 0
    X, y = X[keep], y[keep]

    print('\n=== feature means: TP vs FP (all dets) ===')
    print(f'{"feature":<9}{"TP":<10}{"FP":<10}')
    for j, n in enumerate(names):
        print(f'{n:<9}{X[y == 0, j].mean():<10.3f}{X[y == 1, j].mean():<10.3f}')
    print('\n=== feature means in the rescue zone [0.3,0.85) ===')
    m = (X[:, -1] >= 0.3) & (X[:, -1] < 0.85)
    for j, n in enumerate(names):
        print(f'{n:<9}{X[m & (y == 0), j].mean():<10.3f}{X[m & (y == 1), j].mean():<10.3f}')

    # held-out split by image
    rel_list = sorted(per_rel)
    test_rels = set(rel_list[1::2])
    tr = np.concatenate([np.full(len(per_rel[r][2]), r not in test_rels) for r in rel_list])
    tr = tr[keep]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[~tr], y[~tr]

    auc_score = auc(Xte[:, -1], yte == 0)
    w, b, mu, sd = logistic_fit(Xtr, (ytr == 0).astype(np.float32))
    logit = (Xte - mu) / sd @ w.numpy() + b.item()
    auc_full = auc(logit, yte == 0)
    print(f'\nheld-out AUC: score alone = {auc_score:.4f}, score+profile = {auc_full:.4f}')

    print('\n=== AP on held-out half: baseline vs rescaled by logistic ===')
    base = {r: per_rel[r][0] for r in test_rels}
    res = wider_ap(base, rels=list(test_rels))
    print('baseline        ', '  '.join(f'{t}={res[t][0]:.4f}' for t in sorted(res)), flush=True)
    re = {}
    for r in test_rels:
        det, feats, _ = per_rel[r]
        if len(det):
            s = 1 / (1 + np.exp(-(((feats - mu) / sd @ w.numpy()) + b.item())))
            det = det.copy()
            det[:, 4] = s
        re[r] = det
    res = wider_ap(re, rels=list(test_rels))
    print('profile-rescaled', '  '.join(f'{t}={res[t][0]:.4f}' for t in sorted(res)), flush=True)


if __name__ == '__main__':
    main()
