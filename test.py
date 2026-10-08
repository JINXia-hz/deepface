import torch
import load_data
from net import Net


def evaluate(net, tag, loader):
    net.eval()
    correct = 0
    total = 0
    TP = TN = FP = FN = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    print(f"[{tag}] Accuracy: {accuracy:.4f}")
    print(f"[{tag}] TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")


if __name__ == '__main__':
    _, _, test_loader = load_data.get_data_loaders(batch_size=128)
    net = Net()
    net.load_state_dict(torch.load('models/model_noeq.pth'))
    evaluate(net, 'final test', test_loader)
