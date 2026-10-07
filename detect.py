import sys                                                                            
from PIL import Image, ImageDraw                                                      
import torch
import torchvision
import torchvision.transforms as transforms
from net import Net                                           
                                                                                           
base_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def equalize_patches(patches):
    # Vectorized reimplementation of ImageOps.equalize (per window), followed by Normalize(0.5, 0.5)
    # patches: [N, 1, 36, 36], value range [0, 1]
    N = patches.size(0)
    q = (patches * 255).round().long().reshape(N, -1)
    hist = torch.zeros(N, 256, dtype=torch.long)
    hist.scatter_add_(1, q, torch.ones_like(q))
    nonzero = hist != 0
    last_idx = 255 - nonzero.flip(1).long().argmax(1)   # last non-zero bin
    last_cnt = hist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
    step = (hist.sum(1) - last_cnt) // 255
    identity = (nonzero.sum(1) <= 1) | (step == 0)      # degenerate window: identity mapping
    step = step.clamp(min=1).unsqueeze(1)
    csum = hist.cumsum(1) - hist                        # exclusive prefix sum
    lut = ((step // 2 + csum) // step).clamp(max=255)
    lut = torch.where(identity.unsqueeze(1), torch.arange(256).expand(N, 256), lut)
    out = lut.gather(1, q).float() / 255.0
    return (out.view_as(patches) - 0.5) / 0.5

STRIDE = 8
THRESHOLD = 0.995
RATIO = 1.25
BATCH_SIZE = 2048

img = Image.open(sys.argv[1])                                                         

scale = 1.0                                                                           
pyramid = []                              
w, h = img.size

# build image pyramid
while min(w, h) >= 36:                                                                
    resized = img.resize((int(w), int(h)), Image.BILINEAR)                            
    pyramid.append((base_transform(resized), scale))                                       
    w, h = w / RATIO, h / RATIO                                                         
    scale *= RATIO

# load the pre-trained model
net = Net()
net.load_state_dict(torch.load('model.pth'))
net.eval()
boxes = [] # list of detected boxes

# run inference on each patch
for i, (t, scale) in enumerate(pyramid):
    t_patches = t.unfold(1, 36, STRIDE).unfold(2, 36, STRIDE)
    H_n, W_n = t_patches.shape[1], t_patches.shape[2] 
    patches = t_patches.reshape(-1, 1, 36, 36)
    rows = torch.arange(H_n) * STRIDE
    cols = torch.arange(W_n) * STRIDE
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
    xs = grid_x.reshape(-1)                          
    ys = grid_y.reshape(-1)
    probs = []
    with torch.no_grad():
        for s in range(0, patches.size(0), BATCH_SIZE):
            chunk = equalize_patches(patches[s:s + BATCH_SIZE])
            probs.append(torch.softmax(net(chunk), dim=1)[:, 1])
    probs = torch.cat(probs) if probs else torch.empty(0)
    keep = probs >= THRESHOLD                                                             
                            
    for x, y, p in zip(xs[keep], ys[keep], probs[keep]):                                                                                                    
        boxes.append((x.item()*scale, y.item()*scale, (x.item()+36)*scale, (y.item()+36)*scale, p.item()))

# apply non-maximum suppression
if boxes:
    boxes = torch.tensor(boxes)
    keep = torchvision.ops.nms(boxes[:, :4], boxes[:, 4], 0.2)
    boxes = boxes[keep]

# draw the detected boxes on the original image
draw = ImageDraw.Draw(img)
for x1, y1, x2, y2, _ in boxes:
    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
import time
for attempt in range(5):
    try:
        img.save('detect_output.png')
        break
    except OSError:
        time.sleep(1)
else:
    img.save('detect_output_alt.png')
print(f"detected {len(boxes)} faces")