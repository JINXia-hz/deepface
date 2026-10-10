import os
import sys

import torch
from PIL import Image

from deepface.detect import BACKENDS, SCANNERS
from deepface.evaluate import eval_jobs
from deepface.models import load_model

torch.set_num_threads(os.cpu_count() or 1)

MAX_CANDS = 5000  # deep backend only (dense score maps produce many peaks)


def main(model_path, out_path):
    net, backend = load_model(model_path)
    cfg = BACKENDS[backend]

    jobs = eval_jobs()
    data = {}
    for i, (group, path) in enumerate(jobs):
        c = SCANNERS[backend](net, Image.open(path), cfg['floor'], cfg['ratio'])
        if backend == 'deep' and len(c) > MAX_CANDS:
            c = c[c[:, 4].argsort(descending=True)][:MAX_CANDS]
        data[(group, path if group == 'wider' else os.path.basename(path))] = c
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(jobs)}", flush=True)
    torch.save(data, out_path)
    print(f"saved {len(data)} images -> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
