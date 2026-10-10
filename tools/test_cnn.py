import torch

from deepface.data import get_data_loaders
from deepface.evaluate import evaluate
from deepface.models import Net

if __name__ == '__main__':
    _, _, test_loader = get_data_loaders(batch_size=128)
    net = Net()
    net.load_state_dict(torch.load('models/model_noeq.pth', weights_only=True))
    evaluate(net, 'final test', test_loader)
