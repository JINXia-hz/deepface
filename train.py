import load_data
import torch
import torch.optim as optim
import torch.nn as nn
from net import Net
from test import evaluate

train_loader, valid_loader, _ = load_data.get_data_loaders(batch_size=128, valid_size=0.2)

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
    train(net, train_loader, valid_loader, criterion, epochs=2)
    torch.save(net.state_dict(), 'model.pth')
