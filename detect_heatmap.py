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

def nms_iom(boxes, scores, thresh=0.5):                                               
    order = scores.argsort(descending=True)                                           
    keep = []                                                                         
    while order.numel() > 0:                                                          
        i = order[0]                                                                  
        keep.append(i)                                                                
        rest = order[1:]                                                              
        if rest.numel() == 0:                                                         
            break                                                                     
        xx1 = torch.maximum(boxes[i, 0], boxes[rest, 0])                              
        yy1 = torch.maximum(boxes[i, 1], boxes[rest, 1])                              
        xx2 = torch.minimum(boxes[i, 2], boxes[rest, 2])                              
        yy2 = torch.minimum(boxes[i, 3], boxes[rest, 3])                              
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)                   
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])            
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iom = inter / torch.minimum(area_i, area_r)                                   
        order = rest[iom <= thresh]                                                   
    return keep

DOWNSAMPLE = 2 ** 3
THRESHOLD = 0.995
RATIO = 1.05

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
net.load_state_dict(torch.load('models/model_noeq.pth'))
net.eval()
fc_w = net.fc.weight.view(2,64,1,1)
fc_b = net.fc.bias
boxes = [] # list of detected boxes

# run full-convolutional inference on each scale of the pyramid
for t, scale in pyramid:
    with torch.no_grad():
        f = net.features(t.unsqueeze(0))  # [1, 64, H', W']
        f = F.avg_pool2d(f, kernel_size=4, stride=1)
        logits = F.conv2d(f, fc_w, fc_b)
        heatmap = torch.softmax(logits, dim=1)[0, 1]  # [H', W']
        ys, xs = torch.nonzero(heatmap >= THRESHOLD, as_tuple=True)
        scores = heatmap[ys, xs]
        for x, y, p in zip(xs, ys, scores):
            boxes.append((x.item()*DOWNSAMPLE*scale, y.item()*DOWNSAMPLE*scale, (x.item()*DOWNSAMPLE+36)*scale, (y.item()*DOWNSAMPLE+36)*scale, p.item()))

if boxes:
    boxes = torch.tensor(boxes)
    keep = nms_iom(boxes[:, :4], boxes[:, 4], 0.2)
    boxes = boxes[torch.stack(keep)]

# draw the detected boxes on the original image
draw = ImageDraw.Draw(img)
for x1, y1, x2, y2, _ in boxes:
    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
import os
name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
img.save(f'outputs/{name}_{len(boxes)}faces.png')
print(f"detected {len(boxes)} faces")