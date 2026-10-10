import os
import sys

from PIL import Image, ImageDraw

from deepface.detect import detect
from deepface.models import load_model


def main():
    img_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/model_deep.pth'
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else None

    img = Image.open(img_path)
    net, backend = load_model(model_path)
    boxes = detect(net, img, backend, thr)

    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, _ in boxes:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
    name = os.path.splitext(os.path.basename(img_path))[0]
    img.save(f'outputs/{name}_{len(boxes)}faces.png')
    print(f"[{backend}] detected {len(boxes)} faces -> outputs/{name}_{len(boxes)}faces.png")


if __name__ == '__main__':
    main()
