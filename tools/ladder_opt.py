"""Optimize the pyramid ladder itself: non-uniform level placement.

The ladder is a sampling strategy of the log-scale axis. Given a compute
budget (sum of s_k^2 over kept levels), the optimal placement minimizes the
interpolation loss sum_k rho(k) * kappa(k) * d_k^2, where rho = density of
matched faces peaking at level k, kappa = mean log-score profile curvature,
d_k = distance to the nearest kept level. Solved exactly by segment DP over
the 31 cached ratio-1.1 levels, then validated with real AP + scale
interpolation on the 244-image sample at equal budget to the uniform 1.21
ladder.

Usage: py -m tools.ladder_opt
"""
import numpy as np
import torch

from deepface.detect import cluster_nms
from deepface.evaluate import load_val_gt, wider_ap
from tools.eval_wider import sample_rels
from tools.level_redundancy import iou_matrix, load_cands
from tools.level_support import match_gt

RATIO = 1.1
N_LEVELS = 31


def collect_stats(rels, boxes, valid):
    """rho[k]: matched-TP count by peak level; kappa[k]: mean |curvature|."""
    rho = np.zeros(N_LEVELS)
    kap_sum = np.zeros(N_LEVELS)
    kap_n = np.zeros(N_LEVELS)
    dets_cache = {}
    for i, rel in enumerate(rels):
        r = load_cands(rel)
        if r is None:
            continue
        c, levels = r
        t = torch.from_numpy(c)
        det = cluster_nms(t[t[:, 4].argsort(descending=True)], 0.05, 'std').numpy()
        dets_cache[rel] = (c, levels, det)
        if len(det) == 0:
            continue
        lab = match_gt(det, boxes.get(rel, np.zeros((0, 4))), valid[rel])
        for d, tp in zip(det, lab):
            if tp != 0:
                continue
            size = d[2] - d[0]
            k = round(np.log(size / 36) / np.log(RATIO))
            if not (0 <= k < N_LEVELS):
                continue
            rho[k] += 1
            ys = [np.log(max(d[4], 1e-6))]
            ok = True
            for dk in (-1, 1):
                m = levels == k + dk
                if not m.any():
                    ok = False
                    break
                ious = iou_matrix(d[None, :4], c[m, :4])[0]
                j = int(ious.argmax())
                if ious[j] < 0.2:
                    ok = False
                    break
                ys.append(np.log(max(c[m][j, 4], 1e-6)))
            if ok and len(ys) == 3:
                kap = abs((ys[1] + ys[2]) / 2 - ys[0])
                kap_sum[k] += kap
                kap_n[k] += 1
        if (i + 1) % 50 == 0:
            print(f'stats {i + 1}/{len(rels)}', flush=True)
    kappa = np.where(kap_n > 0, kap_sum / np.maximum(kap_n, 1), 0)
    kappa[kap_n == 0] = kappa[kap_n > 0].mean() if kap_n.any() else 1.0
    return rho, kappa, dets_cache


def optimize_ladder(rho, kappa, budget, n_levels=N_LEVELS):
    """DP: choose kept levels (must include 0 and n_levels-1) minimizing
    total interpolation loss, s.t. sum of 1.1^(-2k) over kept <= budget."""
    w = rho * kappa
    cost = RATIO ** (-2 * np.arange(n_levels))
    # segment loss L[i][j]: levels strictly between kept i and j
    L = np.zeros((n_levels, n_levels))
    for i in range(n_levels):
        for j in range(i + 1, n_levels):
            ks = np.arange(i + 1, j)
            L[i, j] = (w[i + 1:j] * np.minimum(ks - i, j - ks) ** 2).sum()
    B = round(budget / cost[-1])            # quantize by cheapest level
    c_q = np.maximum(np.round(cost / cost[-1]).astype(int), 1)
    INF = 1e18
    dp = np.full((n_levels, B + 1), INF)
    par = np.full((n_levels, B + 1), -1, dtype=int)
    dp[0, c_q[0]] = 0.0
    for j in range(1, n_levels):
        for i in range(j):
            b_new = c_q[i] + c_q[j]
            if b_new > B:
                continue
            src = dp[i, :B - c_q[j] + 1] if c_q[j] <= B else None
            if src is None:
                continue
            cand = dp[i, :B + 1 - c_q[j]] + L[i, j]
            tgt = dp[j, c_q[j]:]
            better = cand < tgt[:len(cand)]
            tgt[:len(cand)][better] = cand[better]
            par[j, c_q[j]:len(cand) + c_q[j]][better] = i
    j = n_levels - 1
    b = int(dp[j].argmin())
    if not np.isfinite(dp[j, b]):
        raise RuntimeError('budget infeasible')
    kept = [j]
    while j > 0:
        i = par[j, b]
        kept.append(i)
        b -= c_q[j]
        j = i
    return sorted(kept)


def ladder_dets(c, levels, kept, interp):
    """NMS on kept levels + optional parabola refinement across kept gaps."""
    m = np.isin(levels, kept)
    ck, lk = c[m], levels[m]
    if len(ck) == 0:
        return ck
    t = torch.from_numpy(ck)
    det = cluster_nms(t[t[:, 4].argsort(descending=True)], 0.05, 'std').numpy()
    if not interp or len(det) == 0:
        return det
    kept_arr = np.array(sorted(kept))
    out = det.copy()
    for i, d in enumerate(det):
        size = d[2] - d[0]
        k = round(np.log(size / 36) / np.log(RATIO))
        below = kept_arr[kept_arr < k]
        above = kept_arr[kept_arr > k]
        if len(below) == 0 or len(above) == 0:
            continue
        pts = [(0.0, d[4])]
        for kk in (below[-1], above[0]):
            mm = lk == kk
            if not mm.any():
                pts = None
                break
            ious = iou_matrix(d[None, :4], ck[mm, :4])[0]
            j = int(ious.argmax())
            if ious[j] < 0.2:
                pts = None
                break
            pts.append((float(kk - k), ck[mm][j, 4]))
        if pts is None:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.log(np.maximum([p[1] for p in pts], 1e-6))
        a, b, _ = np.polyfit(xs, ys, 2)
        if a >= 0:
            continue
        lo, hi = below[-1] - k, above[0] - k
        x_star = float(np.clip(-b / (2 * a), lo, hi))
        cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
        ns = 36 * RATIO ** (k + x_star)
        out[i] = [cx - ns / 2, cy - ns / 2, cx + ns / 2, cy + ns / 2, d[4]]
    return out


def main():
    rels, boxes, tier_masks = load_val_gt()
    rels = sample_rels(rels, 4)
    valid = {r: np.zeros(len(boxes[r]), dtype=bool) for r in rels}
    for masks in tier_masks.values():
        for r, m in masks.items():
            if r in valid:
                valid[r] |= m

    rho, kappa, dets_cache = collect_stats(rels, boxes, valid)
    print('\nlevel: rho (matched faces) / kappa (curvature)')
    for k in range(N_LEVELS):
        if rho[k] > 0:
            print(f'  {k:2d}: rho={rho[k]:5.0f} kappa={kappa[k]:.3f}')

    cost = RATIO ** (-2 * np.arange(N_LEVELS))
    even = list(range(0, N_LEVELS, 2))
    budget = cost[even].sum()
    print(f'\nbudget (even ladder cost) = {budget:.3f} of full {cost.sum():.3f}')

    kept = optimize_ladder(rho, kappa, budget)
    print(f'optimal ladder: {kept} (cost {cost[kept].sum():.3f})')
    gaps = np.diff(kept)
    print(f'gaps: {gaps.tolist()}')

    ladders = [('uniform 1.21 + interp', even), ('optimized + interp', kept)]
    for name, ks in ladders:
        dets = {rel: ladder_dets(c, lv, ks, True)
                for rel, (c, lv, _) in dets_cache.items()}
        res = wider_ap(dets, rels=list(dets))
        print(f'{name:26s}', '  '.join(f'{t}={res[t][0]:.4f}' for t in sorted(res)),
              flush=True)


if __name__ == '__main__':
    main()
