from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def MNIST(path="./data/ds", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Загружаем датасеты
    train_dataset = datasets.MNIST(
        root=path,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=path,
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
