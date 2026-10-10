import math
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from deepface.detect import GRAY
from deepface.models import Net
from deepface.wider import load_annotations

torch.set_num_threads(os.cpu_count() or 1)
torch.manual_seed(0)

CANDS_PATH = 'outputs/cache/cands_fcn.pt'
KERNEL_OUT = 'outputs/cache/kernel.pt'


def main(model_path='models/model_fcn.pth'):
    net = Net()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()
    fc_w = net.fc.weight.view(2, 64, 1, 1)

    ann = load_annotations('data/widerface/wider_face_split/wider_face_val_bbx_gt.txt')

    # ---------- 1) importance kernel K (60x60, offsets -14..+45 rel. to window top-left)
    K = torch.zeros(60, 60)
    count = 0
    for rel in sorted(ann)[200:700]:
        img = Image.open(os.path.join('data/widerface/WIDER_val/images', rel)).convert('L')
        x = GRAY(img).unsqueeze(0).requires_grad_(True)
        f = net.features(x)
        f = F.avg_pool2d(f, kernel_size=4, stride=1)
        logits = F.conv2d(f, fc_w, net.fc.bias)
        logp = torch.log_softmax(logits, dim=1)[0, 1]
        prob = logp.exp()
        ys, xs = torch.nonzero(prob >= 0.995, as_tuple=True)
        sel = []
        for y, xx in zip(ys.tolist(), xs.tolist()):
            if all(abs(y - y0) + abs(xx - x0) >= 6 for y0, x0 in sel):
                sel.append((y, xx))
        for y, xx in sel[:3]:
            if x.grad is not None:
                x.grad.zero_()
            logp[y, xx].backward(retain_graph=True)
            g = x.grad[0, 0].abs()
            wy, wx = 8 * y, 8 * xx
            region = g[wy - 14:wy + 46, wx - 14:wx + 46]
            if region.shape == (60, 60):
                K += region
                count += 1
        if count >= 400:
            break
    K /= count
    print(f"kernel from {count} detections")

    # g(d): normalized overlap of K with its shifted copy ("soft IoM" shape)
    def g(dy, dx):
        a = K[max(0, dy):60 + min(0, dy), max(0, dx):60 + min(0, dx)]
        b = K[max(0, -dy):60 + min(0, -dy), max(0, dx):60 + min(0, -dx)]
        return torch.minimum(a, b).sum().item() / K.sum().item()

    print("g_r (soft-IoM) vs displacement r (window px):")
    for r in range(0, 45, 4):
        vals = [g(int(dy), int(dx)) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                if abs((dy * dy + dx * dx) ** 0.5 - r) < 2]
        if vals:
            print(f"  r={r:2d}: {sum(vals) / len(vals):.3f}")

    kn = (K / K.max() * 255).byte().numpy()
    Image.fromarray(kn).resize((240, 240), Image.NEAREST).save('outputs/importance_kernel.png')
    torch.save(K, KERNEL_OUT)

    # ---------- 2) response envelope around isolated peaks (from cached candidates)
    data = torch.load(CANDS_PATH, weights_only=False)
    bins = torch.zeros(12)   # distance bins of 4 window-px, 0..44
    strong_cnt = torch.zeros(12)
    peaks = 0
    for (group, key), v in data.items():
        if group != 'wider' or len(v) == 0:
            continue
        strong = v[v[:, 4] >= 0.995]
        for s in strong:
            sc = (s[:2] + s[2:4]) / 2
            unit = (s[2] - s[0]) / 36
            d_strong = (((strong[:, :2] + strong[:, 2:4]) / 2 - sc) ** 2).sum(1).sqrt() / unit
            if (d_strong[:-1] < 48).sum() > 1:  # not isolated
                continue
            peaks += 1
            d_all = (((v[:, :2] + v[:, 2:4]) / 2 - sc) ** 2).sum(1).sqrt() / unit
            near = v[d_all < 48]
            d_near = d_all[d_all < 48]
            b = (d_near / 4).long().clamp(max=11)
            bins += torch.bincount(b, minlength=12)
            strong_cnt += torch.bincount(b[near[:, 4] >= 0.995], minlength=12)
    print(f"\nresponse envelope around {peaks} isolated peaks (dist in window px):")
    for i in range(12):
        if bins[i] > 0:
            print(f"  d={i * 4:2d}..{i * 4 + 4:2d}: candidates {int(bins[i]):5d}, "
                  f"of which >=0.995: {strong_cnt[i] / bins[i]:.3f}")

    # ---------- 3) nearest-neighbor distance between distinct GT faces (must-not-merge)
    nnd = []
    for rel in sorted(ann)[:200]:
        boxes = ann[rel][:, :4]
        keep = [b for b in boxes if min(b[2], b[3]) >= 28]
        for i in range(len(keep)):
            xi, yi, wi, hi = keep[i]
            ci = (xi + wi / 2, yi + hi / 2)
            for j in range(i + 1, len(keep)):
                xj, yj, wj, hj = keep[j]
                cj = (xj + wj / 2, yj + hj / 2)
                unit = max(wi, hi, wj, hj) / 36  # window units of the larger face
                nnd.append(math.hypot(ci[0] - cj[0], ci[1] - cj[1]) / unit)
    nnd.sort()
    n = len(nnd)
    print(f"\nGT distinct-face center distances (window px, n={n}):")
    for q in (0.01, 0.05, 0.10, 0.25, 0.50):
        print(f"  {int(q * 100):2d}% quantile: {nnd[int(q * n)]:.1f}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'models/model_fcn.pth')
