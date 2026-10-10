import os
import random

import numpy as np
import torch
from PIL import Image

WINDOW = 36          # classifier window size
DOWNSAMPLE = 8       # 3 maxpools
POS_MIN, POS_MAX = 26, 52   # a 36px window is a positive only for faces of this scaled size

POS_R = 6   # deep backend: positive disk radius around each in-range face center (tile px)
BORDER = 8  # deep backend: tile border band labeled ignore (partial context at the edge)


def load_annotations(txt_path):
    """Parse wider_face_*_bbx_gt.txt -> {rel_path: [N,10] float32}.

    Box columns: x, y, w, h, blur, expression, illumination, invalid, occlusion, pose.
    """
    ann = {}
    with open(txt_path) as f:
        lines = [l.strip() for l in f]
    i = 0
    while i < len(lines):
        path = lines[i]; i += 1
        n = int(lines[i]); i += 1
        boxes = np.zeros((n, 10), dtype=np.float32)
        for k in range(n):
            boxes[k] = [float(v) for v in lines[i].split()[:10]]
            i += 1
        if n == 0 and i < len(lines):
            # the train annotation file emits a dummy all-zero box line for
            # face-free images; consume it if present
            try:
                [float(v) for v in lines[i].split()]
                i += 1
            except ValueError:
                pass
        ann[path] = boxes
    return ann


def heatmap_labels(boxes, any_mask, pos_mask, hw):
    """Label each heatmap position of a tile (heatmap backend, stride 8).

    boxes: [M,4] (x1,y1,x2,y2) in tile coords. any_mask: boxes that may contain a
    real face (ignore zone around them). pos_mask: boxes eligible to be positives.
    Heatmap position (a,b) corresponds to the 36x36 window with top-left (8b, 8a).
    Returns [hw,hw] long: 1 positive, 0 negative, -1 ignore.
    """
    g = torch.arange(hw, dtype=torch.float32) * DOWNSAMPLE
    wy1 = g.view(-1, 1).expand(hw, hw)
    wx1 = g.view(1, -1).expand(hw, hw)
    wx2, wy2 = wx1 + WINDOW, wy1 + WINDOW
    labels = torch.zeros(hw, hw, dtype=torch.long)
    if len(boxes) == 0:
        return labels

    b = torch.as_tensor(boxes, dtype=torch.float32)
    ix1 = torch.maximum(wx1.unsqueeze(-1), b[None, None, :, 0])
    iy1 = torch.maximum(wy1.unsqueeze(-1), b[None, None, :, 1])
    ix2 = torch.minimum(wx2.unsqueeze(-1), b[None, None, :, 2])
    iy2 = torch.minimum(wy2.unsqueeze(-1), b[None, None, :, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)   # [hw,hw,M]
    box_area = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    iou = inter / (WINDOW * WINDOW + box_area[None, None, :] - inter + 1e-6)
    iow = inter / (WINDOW * WINDOW)

    any_mask = torch.as_tensor(any_mask, dtype=torch.bool)
    pos_mask = torch.as_tensor(pos_mask, dtype=torch.bool)
    if any_mask.any():
        labels[iow[..., any_mask].max(dim=-1).values >= 0.25] = -1
    if pos_mask.any():
        labels[iou[..., pos_mask].max(dim=-1).values >= 0.5] = 1
    # border windows reach past the tile edge into padding; skip them
    labels[-2:, :] = -1
    labels[:, -2:] = -1
    return labels


def disk_labels(boxes, any_mask, pos_mask, tile):
    """Stride-1 labels for a tile (deep backend): 1 on a small disk at each
    in-range face center, -1 (ignore) inside every plausibly-face disk,
    0 elsewhere.

    boxes: [M,4] (x1,y1,x2,y2) in tile coords. Returns [tile,tile] long.
    """
    labels = torch.zeros(tile, tile, dtype=torch.long)
    if len(boxes):
        g = torch.arange(tile, dtype=torch.float32)
        yy = g.view(-1, 1).expand(tile, tile)
        xx = g.view(1, -1).expand(tile, tile)
        b = torch.as_tensor(boxes, dtype=torch.float32)
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        side = torch.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        d2 = (xx.unsqueeze(-1) - cx) ** 2 + (yy.unsqueeze(-1) - cy) ** 2
        any_mask = torch.as_tensor(any_mask, dtype=torch.bool)
        pos_mask = torch.as_tensor(pos_mask, dtype=torch.bool)
        if any_mask.any():
            r = (side[any_mask] / 2).clamp(min=POS_R + 2)
            labels[(d2[..., any_mask] <= r ** 2).any(dim=-1)] = -1
        if pos_mask.any():
            labels[(d2[..., pos_mask] <= POS_R ** 2).any(dim=-1)] = 1
    labels[:BORDER, :] = -1
    labels[-BORDER:, :] = -1
    labels[:, :BORDER] = -1
    labels[:, -BORDER:] = -1
    return labels


class SceneSampler:
    """Random grayscale tiles from WIDER FACE with per-position heatmap labels.

    70% of tiles are centered near a random face rescaled to 30-48px (the size the
    36px classifier window actually fires on); 30% are plain random crops.
    """

    def __init__(self, img_root, txt_path, tile=192, skip_images=0):
        self.img_root = img_root
        self.tile = tile
        ann = load_annotations(txt_path)
        self.items = []
        for rel in sorted(ann):
            boxes = ann[rel]
            good = [k for k in range(len(boxes))
                    if boxes[k, 7] == 0 and min(boxes[k, 2], boxes[k, 3]) >= 12]
            self.items.append((rel, boxes, good))
        self.items = self.items[skip_images:]
        self.with_face = [it for it in self.items if it[2]]
        self.hw = tile // DOWNSAMPLE - 3
        self.label_shape = (self.hw, self.hw)

    def make_labels(self, tb, any_mask, pos_mask):
        return heatmap_labels(tb, any_mask, pos_mask, self.hw)

    def sample(self):
        rel, boxes, _ = (random.choice(self.with_face) if random.random() < 0.7
                         else random.choice(self.items))
        im = Image.open(os.path.join(self.img_root, rel)).convert('L')
        w, h = im.size

        anchor = None
        face_idx = [k for k in range(len(boxes))
                    if boxes[k, 7] == 0 and min(boxes[k, 2], boxes[k, 3]) >= 12]
        if face_idx and random.random() < 0.7:
            anchor = boxes[random.choice(face_idx)]
            scale = random.uniform(30, 48) / min(anchor[2], anchor[3])
        else:
            scale = random.uniform(0.5, 2.0)
        scale = max(scale, (self.tile + 1) / w, (self.tile + 1) / h)
        sw, sh = int(w * scale), int(h * scale)
        im = im.resize((sw, sh), Image.BILINEAR)

        if anchor is not None:
            cx, cy = (anchor[0] + anchor[2] / 2) * scale, (anchor[1] + anchor[3] / 2) * scale
            ox = int(cx - self.tile / 2 + random.uniform(-self.tile / 4, self.tile / 4))
            oy = int(cy - self.tile / 2 + random.uniform(-self.tile / 4, self.tile / 4))
        else:
            ox = random.randint(0, sw - self.tile)
            oy = random.randint(0, sh - self.tile)
        ox = min(max(ox, 0), sw - self.tile)
        oy = min(max(oy, 0), sh - self.tile)

        arr = np.array(im.crop((ox, oy, ox + self.tile, oy + self.tile)), dtype=np.uint8)

        tb, any_mask, pos_mask = [], [], []
        for k in range(len(boxes)):
            x, y, bw, bh = boxes[k, :4] * scale
            x1, y1 = x - ox, y - oy
            x2, y2 = x1 + bw, y1 + bh
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, self.tile), min(y2, self.tile)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            tb.append((x1, y1, x2, y2))
            side = min(x2 - x1, y2 - y1)
            any_mask.append(boxes[k, 7] == 0 or side >= 8)
            pos_mask.append(boxes[k, 7] == 0 and POS_MIN <= side <= POS_MAX)

        if random.random() < 0.5:
            arr = arr[:, ::-1].copy()
            tb = [(self.tile - x2, y1, self.tile - x1, y2) for x1, y1, x2, y2 in tb]

        x = torch.from_numpy(arr).float().mul_(2.0 / 255.0).sub_(1.0)
        labels = self.make_labels(tb, any_mask, pos_mask)
        return x, labels

    def sample_batch(self, n):
        xs, ls = zip(*(self.sample() for _ in range(n)))
        return torch.stack(xs).unsqueeze(1), torch.stack(ls)


class NegativeSampler:
    """Random tiles from face-free scene images (COCO no_person); all-negative labels."""

    def __init__(self, root, tile=192):
        self.paths = sorted(glob_jpgs(root))
        self.tile = tile
        self.hw = tile // DOWNSAMPLE - 3
        self.label_shape = (self.hw, self.hw)

    def sample(self):
        im = Image.open(random.choice(self.paths)).convert('L')
        w, h = im.size
        scale = max(random.uniform(0.5, 2.0), (self.tile + 1) / w, (self.tile + 1) / h)
        sw, sh = int(w * scale), int(h * scale)
        im = im.resize((sw, sh), Image.BILINEAR)
        ox = random.randint(0, sw - self.tile)
        oy = random.randint(0, sh - self.tile)
        arr = np.array(im.crop((ox, oy, ox + self.tile, oy + self.tile)), dtype=np.uint8)
        if random.random() < 0.5:
            arr = arr[:, ::-1].copy()
        x = torch.from_numpy(arr).float().mul_(2.0 / 255.0).sub_(1.0)
        return x, torch.zeros(*self.label_shape, dtype=torch.long)

    def sample_batch(self, n):
        xs, ls = zip(*(self.sample() for _ in range(n)))
        return torch.stack(xs).unsqueeze(1), torch.stack(ls)


class DeepSceneSampler(SceneSampler):
    """SceneSampler with stride-1 center-disk labels instead of stride-8 windows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_shape = (self.tile, self.tile)

    def make_labels(self, tb, any_mask, pos_mask):
        return disk_labels(tb, any_mask, pos_mask, self.tile)


class DeepNegativeSampler(NegativeSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_shape = (self.tile, self.tile)


def glob_jpgs(root):
    return [os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
