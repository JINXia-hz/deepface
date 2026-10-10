import os
import sys

import torch
import torch.optim as optim
import torch.nn as nn

from deepface.data import get_data_loaders
from deepface.models import Net
from deepface.evaluate import evaluate

torch.set_num_threads(os.cpu_count() or 1)

train_loader, valid_loader, _ = get_data_loaders(batch_size=256, valid_size=0.2)

net = Net()
criterion = nn.CrossEntropyLoss(reduction='none')


def train(net, trainloader, validloader, criterion, lr=0.01, momentum=0.9, epochs=10):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):

        net.train()

        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            faceloss = loss[labels == 1].mean()
            nonfaceloss = loss[labels == 0].mean()
            total_loss = torch.nan_to_num(faceloss, nan=0.0) + torch.nan_to_num(nonfaceloss, nan=0.0)
            total_loss.backward()
            optimizer.step()
            if i % 200 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {i}/{len(trainloader)}, Loss: {total_loss.item():.4f}")

        evaluate(net, f"epoch {epoch + 1} valid", validloader)


if __name__ == '__main__':
    train(net, train_loader, valid_loader, criterion, epochs=int(sys.argv[1]))
    torch.save(net.state_dict(), 'models/model_noeq.pth')
