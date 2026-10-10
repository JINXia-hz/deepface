import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from deepface.data import get_data_loaders
from deepface.evaluate import evaluate
from deepface.models import Net
from deepface.wider import SceneSampler, NegativeSampler

torch.set_num_threads(os.cpu_count() or 1)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
LAMBDA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = sys.argv[3] if len(sys.argv) > 3 else 'models/model_fcn2.pth'
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
LR = 1e-4
TILE = 192
N_TRAIN, N_VAL, N_NEG = 10, 2, 4

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

net = Net()
net.load_state_dict(torch.load('models/model_noeq.pth', weights_only=True))
# BN stays in eval mode (running stats) so training matches deployment exactly;
# only conv/fc weights receive gradients.
net.eval()
for p in net.parameters():
    p.requires_grad_(True)

fc_w = net.fc.weight.view(2, 64, 1, 1)


def heatmap_logits(x):
    f = net.features(x)
    f = F.avg_pool2d(f, kernel_size=4, stride=1)
    return F.conv2d(f, fc_w, net.fc.bias)


opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR / 10)

train_loader, valid_loader, _ = get_data_loaders(batch_size=128, valid_size=0.2)
crop_iter = iter(train_loader)

# last 200 val images are held out for scene-level evaluation
scenes_train = SceneSampler('data/widerface/WIDER_train/images',
                            'data/widerface/wider_face_split/wider_face_train_bbx_gt.txt',
                            tile=TILE)
scenes_val = SceneSampler('data/widerface/WIDER_val/images',
                          'data/widerface/wider_face_split/wider_face_val_bbx_gt.txt',
                          tile=TILE, skip_images=200)
neg_scenes = NegativeSampler('data/no_person', tile=TILE)
print(f"wider train: {len(scenes_train.items)}, wider val: {len(scenes_val.items)}, "
      f"no_person: {len(neg_scenes.paths)}")

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

    logits = heatmap_logits(tiles)
    ce = F.cross_entropy(logits, labels, reduction='none', ignore_index=-1)
    scene_loss = (torch.nan_to_num(ce[labels == 1].mean(), nan=0.0)
                  + torch.nan_to_num(ce[labels == 0].mean(), nan=0.0))

    out = net(crops)
    ce2 = F.cross_entropy(out, clab, reduction='none')
    crop_loss = (torch.nan_to_num(ce2[clab == 1].mean(), nan=0.0)
                 + torch.nan_to_num(ce2[clab == 0].mean(), nan=0.0))

    loss = crop_loss + LAMBDA * scene_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

    if step % 50 == 0:
        print(f"step {step}/{STEPS} crop {crop_loss.item():.4f} scene {scene_loss.item():.4f} "
              f"(pos px {(labels == 1).sum().item()})", flush=True)
    if step % 250 == 0:
        evaluate(net, f"step {step} valid", valid_loader)

torch.save(net.state_dict(), OUT)
print(f"saved {OUT}")
