import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

GRAY = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

WINDOW = 36
DOWNSAMPLE = 8       # Net: 3 maxpools

# per-backend defaults: detection threshold, pyramid ratio, candidate-cache floor
BACKENDS = {
    'heatmap': dict(thr=0.995, ratio=1.05, floor=0.85),
    # 1.21 = every other level of the old 1.1 ladder: -0.02 AP for ~2x speed
    # (measured in tools/level_redundancy.py; the [26,52] band is 2x wide)
    'deep': dict(thr=0.85, ratio=1.21, floor=0.05),
}

# box shape calibration (w-scale, h-scale, dy in box-side units): WIDER faces
# are ~1.26x taller than wide (train-set stats); lifts easy AP 0.559 -> 0.643
BOX_CALIBR = (1.0, 1.25, 0.0)


def calibrate_boxes(det, calibr=BOX_CALIBR):
    """Reshape square boxes (side L, center cx,cy) to a*L wide, b*L tall,
    center shifted down by c*L."""
    a, b, c = calibr
    if len(det) == 0 or (a, b, c) == (1.0, 1.0, 0.0):
        return det
    det = det.clone()
    L = det[:, 2] - det[:, 0]
    cx = (det[:, 0] + det[:, 2]) / 2
    cy = (det[:, 1] + det[:, 3]) / 2 + c * L
    det[:, 0], det[:, 2] = cx - a * L / 2, cx + a * L / 2
    det[:, 1], det[:, 3] = cy - b * L / 2, cy + b * L / 2
    return det


def pyramid(img, ratio, min_side=WINDOW, min_scale=1.0, max_scale=None, max_side=4608):
    """Yield (normalized tensor [1,H,W], scale) for each pyramid level.
    scale maps level px -> original px: >1 downscaled, <1 upscaled. Upscale
    levels (min_scale < 1, added for tiny faces) stop below 1.0; levels whose
    resized image exceeds max_side are skipped."""
    W, H = img.size
    scales = []
    s = min_scale
    while s < 1.0 and (max_scale is None or s <= max_scale):
        scales.append(s)
        s *= ratio
    if max_scale is None or max_scale >= 1.0:
        s = 1.0
        while min(W / s, H / s) >= min_side and (max_scale is None or s <= max_scale):
            scales.append(s)
            s *= ratio
    for s in scales:
        w, h = int(W / s), int(H / s)
        if max(w, h) <= max_side:
            yield GRAY(img.resize((w, h), Image.BILINEAR)), s


def scan_heatmap(net, img, thr, ratio=BACKENDS['heatmap']['ratio']):
    """Net backend: fully-convolutional heatmap. Each position >= thr becomes a
    36px window box. Returns [N,5] (x1, y1, x2, y2, score) in original coords."""
    fc_w = net.fc.weight.view(2, 64, 1, 1)
    out = []
    for t, s in pyramid(img, ratio):
        with torch.no_grad():
            f = net.features(t.unsqueeze(0))
            f = F.avg_pool2d(f, kernel_size=4, stride=1)
            heatmap = torch.softmax(F.conv2d(f, fc_w, net.fc.bias), dim=1)[0, 1]
            ys, xs = torch.nonzero(heatmap >= thr, as_tuple=True)
            scores = heatmap[ys, xs]
        for x, y, p in zip(xs.tolist(), ys.tolist(), scores.tolist()):
            out.append((x * DOWNSAMPLE * s, y * DOWNSAMPLE * s,
                        (x * DOWNSAMPLE + WINDOW) * s, (y * DOWNSAMPLE + WINDOW) * s, p))
    return torch.tensor(out) if out else torch.zeros(0, 5)


def _dense_peaks(net, t, thr):
    """3x3 local maxima >= thr on the sigmoid score map of tensor [1,H,W].
    Returns (xs, ys, scores) lists."""
    with torch.no_grad():
        prob = torch.sigmoid(net(t.unsqueeze(0)))[0, 0]
        peaks = (prob == F.max_pool2d(prob[None, None], 3, stride=1, padding=1)[0, 0]) \
                & (prob >= thr)
        ys, xs = torch.nonzero(peaks, as_tuple=True)
        scores = prob[ys, xs]
    return xs.tolist(), ys.tolist(), scores.tolist()


TILE_PX = 2_000_000   # levels larger than this are forwarded in tiles (RAM cap)
TILE, TILE_MARGIN = 896, 64   # tile size and overlap margin (>= receptive field 60)


def _axis_tiles(L, tile, margin):
    """Partition [0,L) into cores of width tile-2*margin; each core's tile is
    the core expanded by margin, clipped to [0,L]."""
    step = tile - 2 * margin
    out = []
    c0 = 0
    while c0 < L:
        c1 = min(c0 + step, L)
        out.append((max(c0 - margin, 0), min(c1 + margin, L), c0, c1))
        c0 = c1
    return out


def _dense_peaks_tiled(net, t, thr):
    """_dense_peaks through an overlapping-tile forward for large levels.
    Each peak is kept only in its tile's core zone, so coverage is exact."""
    _, H, W = t.shape
    if H * W <= TILE_PX:
        return _dense_peaks(net, t, thr)
    xs_all, ys_all, ps_all = [], [], []
    for y0, y1, cy0, cy1 in _axis_tiles(H, TILE, TILE_MARGIN):
        for x0, x1, cx0, cx1 in _axis_tiles(W, TILE, TILE_MARGIN):
            xs, ys, ps = _dense_peaks(net, t[:, y0:y1, x0:x1], thr)
            for x, y, p in zip(xs, ys, ps):
                gx, gy = x + x0, y + y0
                if cx0 <= gx < cx1 and cy0 <= gy < cy1:
                    xs_all.append(gx)
                    ys_all.append(gy)
                    ps_all.append(p)
    return xs_all, ys_all, ps_all


def scan_dense(net, img, thr, ratio=BACKENDS['deep']['ratio'], min_scale=1.0, max_scale=None):
    """ResUNet backend: per-pixel score map. Each 3x3 local maximum >= thr
    becomes a 36px box centered on the peak. Returns [N,5].
    min_scale < 1 adds upscale pyramid levels (for tiny faces)."""
    out = []
    for t, s in pyramid(img, ratio, min_scale=min_scale, max_scale=max_scale):
        xs, ys, ps = _dense_peaks_tiled(net, t, thr)
        for x, y, p in zip(xs, ys, ps):
            out.append(((x - WINDOW / 2) * s, (y - WINDOW / 2) * s,
                        (x + WINDOW / 2) * s, (y + WINDOW / 2) * s, p))
    return torch.tensor(out) if out else torch.zeros(0, 5)


SCANNERS = {'heatmap': scan_heatmap, 'deep': scan_dense}


def build_G(K, R=48):
    """G[dy,dx] = sum(min(K, K shifted by (dy,dx))) / sum(K), offsets in window px."""
    G = torch.zeros(2 * R + 1, 2 * R + 1)
    total = K.sum()
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            a = K[max(0, dy):60 + min(0, dy), max(0, dx):60 + min(0, dx)]
            b = K[max(0, -dy):60 + min(0, -dy), max(0, dx):60 + min(0, -dx)]
            G[dy + R, dx + R] = torch.minimum(a, b).sum() / total
    return G


def cluster_nms(cands, thr, method, iom=0.2, G=None):
    """Greedy cluster NMS. cands: [N,5] sorted by score desc.

    'std'      keep the top box of each cluster, IoM on the 36px core box
    'wavg'     merge cluster by score-weighted average, IoM on the 36px core box
    'dist<R>'  weighted merge; cluster = centers within R window-px of the seed
    'softk'    weighted merge; cluster = soft-IoM of measured kernel > iom (needs G)
    """
    cands = cands[cands[:, 4] >= thr]
    if len(cands) == 0:
        return torch.zeros(0, 5)
    boxes, scores = cands[:, :4], cands[:, 4]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    unit = (boxes[:, 2] - boxes[:, 0]) / WINDOW

    alive = torch.ones(len(cands), dtype=torch.bool)
    out = []
    for i in range(len(cands)):
        if not alive[i]:
            continue
        idx = torch.nonzero(alive, as_tuple=True)[0]
        if method in ('std', 'wavg'):
            ix1 = torch.maximum(boxes[i, 0], boxes[idx, 0])
            iy1 = torch.maximum(boxes[i, 1], boxes[idx, 1])
            ix2 = torch.minimum(boxes[i, 2], boxes[idx, 2])
            iy2 = torch.minimum(boxes[i, 3], boxes[idx, 3])
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
            members = idx[inter / torch.minimum(area_i, area_r) > iom]
        elif method.startswith('dist'):
            R = float(method[4:])
            d = ((cx[idx] - cx[i]) ** 2 + (cy[idx] - cy[i]) ** 2).sqrt() / unit[i]
            members = idx[d < R]
        elif method == 'softk':
            R = (G.shape[0] - 1) // 2
            dy = ((cy[idx] - cy[i]) / unit[i]).round().long().clamp(-R, R)
            dx = ((cx[idx] - cx[i]) / unit[i]).round().long().clamp(-R, R)
            members = idx[G[dy + R, dx + R] > iom]

        if method == 'std':
            out.append(torch.cat([boxes[i], scores[i].view(1)]))
        else:
            p = scores[members]
            merged = (boxes[members] * p[:, None]).sum(dim=0) / p.sum()
            out.append(torch.cat([merged, scores[i].view(1)]))
        alive[members] = False
    return torch.stack(out)


def weighted_nms(boxes, scores, iom=0.2):
    """Greedy clustering by IoM with the top box; merge each cluster into a
    score-weighted average box instead of keeping only the argmax."""
    cands = torch.cat([boxes, scores[:, None]], dim=1)
    cands = cands[scores.argsort(descending=True)]
    return cluster_nms(cands, 0.0, 'wavg', iom)


def _iou_1xn(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def scale_refine(det, cands, ratio):
    """Sub-grid scale refinement: fit a parabola to each detection's log-score
    profile along the pyramid ladder and move the box to the fractional peak
    scale (size 36 * ratio**(k + x*)). Recovers and beats the AP lost to
    ladder thinning — tools/scale_interp.py measured +0.01 AP over the full
    ratio-1.1 ladder at half the forward passes."""
    if len(det) == 0 or len(cands) == 0:
        return det
    cs = cands.numpy()
    levels = np.round(np.log(np.maximum(cs[:, 2] - cs[:, 0], 1) / WINDOW)
                      / np.log(ratio)).astype(int)
    out = det.clone()
    for i in range(len(out)):
        d = out[i].numpy()
        size = max(d[2] - d[0], 1e-6)
        k = int(round(np.log(size / WINDOW) / np.log(ratio)))
        pts = [(0.0, float(d[4]))]
        for dk in (-1, 1):
            m = levels == k + dk
            if not m.any():
                pts = None
                break
            pool = cs[m]
            ious = _iou_1xn(d[:4], pool[:, :4])
            j = int(ious.argmax())
            if ious[j] < 0.2:
                pts = None
                break
            pts.append((float(dk), float(pool[j, 4])))
        if pts is None:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.log(np.maximum([p[1] for p in pts], 1e-6))
        a, b, _ = np.polyfit(xs, ys, 2)
        if a >= 0:
            continue
        x_star = float(np.clip(-b / (2 * a), -1, 1))
        cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
        new_size = WINDOW * ratio ** (k + x_star)
        out[i, 0], out[i, 1] = cx - new_size / 2, cy - new_size / 2
        out[i, 2], out[i, 3] = cx + new_size / 2, cy + new_size / 2
    return out


def detect(net, img, backend, thr=None):
    """Full pipeline: pyramid scan at the backend's defaults + weighted NMS
    + sub-grid scale refinement + box shape calibration.
    Returns [M,5] (x1, y1, x2, y2, score)."""
    cfg = BACKENDS[backend]
    cands = SCANNERS[backend](net, img, thr if thr is not None else cfg['thr'], cfg['ratio'])
    if len(cands) == 0:
        return cands
    det = weighted_nms(cands[:, :4], cands[:, 4], 0.2)
    if backend == 'deep':
        det = scale_refine(det, cands, cfg['ratio'])
    return calibrate_boxes(det)
