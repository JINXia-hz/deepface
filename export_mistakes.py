import torch
import torchvision.utils
import load_data
from net import Net


def save_grid(images, path, nrow):
    grid = torch.stack(images) * 0.5 + 0.5  # denormalize back to [0,1]
    torchvision.utils.save_image(grid, path, nrow=nrow, padding=2)


def main():
    _, _, test_loader = load_data.get_data_loaders(batch_size=128)
    net = Net()
    net.load_state_dict(torch.load('models/model_noeq.pth'))
    net.eval()

    fn_images, fp_images = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            predicted = torch.argmax(net(inputs), dim=1)
            fn_images += list(inputs[(predicted == 0) & (labels == 1)])
            fp_images += list(inputs[(predicted == 1) & (labels == 0)])

    print(f"FN: {len(fn_images)} ")
    print(f"FP: {len(fp_images)} ")

    save_grid(fn_images[:98], 'outputs/mistakes_fn.png', nrow=14)
    if fp_images:
        save_grid(fp_images, 'outputs/mistakes_fp.png', nrow=max(len(fp_images), 1))


if __name__ == '__main__':
    main()
