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
    'deep': dict(thr=0.85, ratio=1.1, floor=0.05),
}


def pyramid(img, ratio, min_side=WINDOW):
    """Yield (normalized tensor [1,H,W], scale) for each pyramid level."""
    scale = 1.0
    w, h = img.size
    while min(w, h) >= min_side:
        yield GRAY(img.resize((int(w), int(h)), Image.BILINEAR)), scale
        w, h = w / ratio, h / ratio
        scale *= ratio


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


def scan_dense(net, img, thr, ratio=BACKENDS['deep']['ratio']):
    """ResUNet backend: per-pixel score map. Each 3x3 local maximum >= thr
    becomes a 36px box centered on the peak. Returns [N,5]."""
    out = []
    for t, s in pyramid(img, ratio):
        with torch.no_grad():
            prob = torch.sigmoid(net(t.unsqueeze(0)))[0, 0]
            peaks = (prob == F.max_pool2d(prob[None, None], 3, stride=1, padding=1)[0, 0]) \
                    & (prob >= thr)
            ys, xs = torch.nonzero(peaks, as_tuple=True)
            scores = prob[ys, xs]
        for x, y, p in zip(xs.tolist(), ys.tolist(), scores.tolist()):
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


def detect(net, img, backend, thr=None):
    """Full pipeline: pyramid scan at the backend's defaults + weighted NMS.
    Returns [M,5] (x1, y1, x2, y2, score)."""
    cfg = BACKENDS[backend]
    cands = SCANNERS[backend](net, img, thr if thr is not None else cfg['thr'], cfg['ratio'])
    if len(cands) == 0:
        return cands
    return weighted_nms(cands[:, :4], cands[:, 4], 0.2)
