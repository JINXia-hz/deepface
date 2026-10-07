import sys                                                                            
from PIL import Image, ImageOps, ImageDraw                                                      
import torch
import torchvision
import torchvision.transforms as transforms
from net import Net                                           
                                                                                           
base_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

window_transform = transforms.Compose([
    transforms.Lambda(ImageOps.equalize),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])
to_pil = transforms.ToPILImage()                                                                                    

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
            chunk = torch.stack([window_transform(to_pil(p)) for p in patches[s:s + BATCH_SIZE]])
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