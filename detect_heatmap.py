import sys                                                                            
from PIL import Image, ImageDraw                                                      
import torch
import torchvision
import torchvision.transforms as transforms
from net import Net
from torch.nn import functional as F                                           
                                                                                           
transform = transforms.Compose(
    [transforms.Grayscale(), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5,),std=(0.5,))])

STRIDE = 8
THRESHOLD = 0.995
RATIO = 1.25

img = Image.open(sys.argv[1])                                                         

scale = 1.0                                                                           
pyramid = []                              
w, h = img.size

# build image pyramid
while min(w, h) >= 36:                                                                
    resized = img.resize((int(w), int(h)), Image.BILINEAR)                            
    pyramid.append((transform(resized), scale))                                       
    w, h = w / RATIO, h / RATIO                                                         
    scale *= RATIO

# load the pre-trained model
net = Net()
net.load_state_dict(torch.load('model_noeq.pth'))
net.eval()
fc_w = net.fc.weight.view(2,64,1,1)
fc_b = net.fc.bias
boxes = [] # list of detected boxes

# run full-convolutional inference on each scale of the pyramid
for t, scale in pyramid:
    with torch.no_grad():
        f = net.features(t.unsqueeze(0))  # [1, 64, H', W']
        f = F.avg_pool2d(f, kernel_size=4)
        logits = F.conv2d(f, fc_w, fc_b)
        heatmap = torch.softmax(logits, dim=1)[0, 1]  # [H', W']
        ys, xs = torch.nonzero(heatmap >= THRESHOLD, as_tuple=True)
        scores = heatmap[ys, xs]
        for x, y, p in zip(xs, ys, scores):
            boxes.append((x.item()*STRIDE*scale, y.item()*STRIDE*scale, (x.item()*STRIDE+36)*scale, (y.item()*STRIDE+36)*scale, p.item()))

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