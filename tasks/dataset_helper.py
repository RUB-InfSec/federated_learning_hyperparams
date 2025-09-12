import argparse

import torch
from torchvision import datasets, transforms


# Function to compute mean and std
def calculate_mean_std(loader):
    total_sum = 0
    total_squared_sum = 0
    total_pixels = 0

    for images, _ in loader:
        # Reshape to (batch_size, num_channels, -1)
        images = images.view(images.size(0), images.size(1), -1)
        total_sum += images.sum(dim=[0, 2])  # Sum over batch and pixels
        total_squared_sum += (images ** 2).sum(dim=[0, 2])  # Sum of squares
        total_pixels += images.size(0) * images.size(2)  # Total pixels per channel

    mean = total_sum / total_pixels
    std = (total_squared_sum / total_pixels - mean ** 2).sqrt()
    return mean, std

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'], help='dataset name')
    args = parser.parse_args()

    basic_transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=basic_transform)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root='./datasets', train=True, download=True, transform=basic_transform)
    elif args.dataset == 'mnist':
        dataset = datasets.MNIST(root='./datasets', train=True, download=True, transform=basic_transform)
    elif args.dataset == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=basic_transform)

    mlaas_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    print(calculate_mean_std(mlaas_loader))
