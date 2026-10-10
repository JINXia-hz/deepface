import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from deepface.data import get_data_loaders
from deepface.evaluate import center_accuracy
from deepface.models import ResUNet
from deepface.wider import DeepSceneSampler, DeepNegativeSampler

torch.set_num_threads(os.cpu_count() or 1)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
LAMBDA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = sys.argv[3] if len(sys.argv) > 3 else 'models/model_deep.pth'
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
LR = 1e-3
TILE = 192
N_TRAIN, N_VAL, N_NEG = 10, 2, 4

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

net = ResUNet(in_channels=1, out_channels=1)
net.train()  # trained from scratch: BN learns its running stats

train_loader, valid_loader, _ = get_data_loaders(batch_size=128, valid_size=0.2)
crop_iter = iter(train_loader)

# last 200 val images are held out for scene-level evaluation
scenes_train = DeepSceneSampler('data/widerface/WIDER_train/images',
                                'data/widerface/wider_face_split/wider_face_train_bbx_gt.txt',
                                tile=TILE)
scenes_val = DeepSceneSampler('data/widerface/WIDER_val/images',
                              'data/widerface/wider_face_split/wider_face_val_bbx_gt.txt',
                              tile=TILE, skip_images=200)
neg_scenes = DeepNegativeSampler('data/no_person', tile=TILE)
print(f"wider train: {len(scenes_train.items)}, wider val: {len(scenes_val.items)}, "
      f"no_person: {len(neg_scenes.paths)}")


def focal_loss(logits, labels):
    """Focal loss (gamma=2) on {-1,0,1} label maps; pos/neg means averaged
    separately so sparse positive disks are not swamped by background."""
    logit = logits.squeeze(1)
    t = labels.clamp(min=0).float()
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, t, reduction='none')
    pt = torch.where(t > 0, p, 1 - p)
    fl = ((1 - pt) ** 2 * ce)[labels >= 0]
    pos = (labels == 1)[labels >= 0]
    return (torch.nan_to_num(fl[pos].mean(), nan=0.0)
            + torch.nan_to_num(fl[~pos].mean(), nan=0.0))


# crop anchor: positive crops get a small center disk, negatives stay all-zero
G36 = torch.arange(36, dtype=torch.float32)
D2 = (G36.view(1, -1) - 17.5) ** 2 + (G36.view(-1, 1) - 17.5) ** 2
CROP_MAP = torch.where(D2 <= 16, torch.tensor(1),
                       torch.where(D2 <= 100, torch.tensor(-1), torch.tensor(0)))


def crop_labels(clab):
    lab = torch.zeros(len(clab), 36, 36, dtype=torch.long)
    lab[clab == 1] = CROP_MAP
    return lab


opt = optim.Adam(net.parameters(), lr=LR)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR / 10)

for step in range(1, STEPS + 1):
    t1, l1 = scenes_train.sample_batch(N_TRAIN)
    t2, l2 = scenes_val.sample_batch(N_VAL)
    nt, nl = neg_scenes.sample_batch(N_NEG)
    tiles = torch.cat([t1, t2, nt])
    labels = torch.cat([l1, l2, nl])
    try:
        crops, clab = next(crop_iter)
    except StopIteration:
        crop_iter = iter(train_loader)
        crops, clab = next(crop_iter)

    scene_loss = focal_loss(net(tiles), labels)
    crop_loss = focal_loss(net(crops), crop_labels(clab))

    loss = crop_loss + LAMBDA * scene_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if step % 50 == 0:
        print(f"step {step}/{STEPS} crop {crop_loss.item():.4f} scene {scene_loss.item():.4f} "
              f"(pos px {(labels == 1).sum().item()})", flush=True)
    if step % 250 == 0:
        center_accuracy(net, f"step {step} valid", valid_loader)

torch.save(net.state_dict(), OUT)
print(f"saved {OUT}")
