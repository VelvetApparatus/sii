import random

import torch
from matplotlib import pyplot as plt
from torch import nn

from db.mnist import MNIST
from models import AutoEncoder
from tqdm import tqdm

from pkg.device import get_device


def add_gaussian_noise(x, mean=0.0, std=0.1):
    noise = torch.randn_like(x) * std + mean
    return torch.clamp(x + noise, 0., 1.)


def train(
        num_epochs: int = 15,
        batch_size: int = 64,
        data_dir: str = "./data",
):
    device = get_device()
    model = AutoEncoder(
        in_channels=1,
        latent_dim=256,
        conv_num=2,
        shape=(28, 28),
    ).to(device)

    train_loader, test_loader = MNIST(path=data_dir, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_model_path = "./best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            noisy_data = add_gaussian_noise(data)

            optimizer.zero_grad()
            output = model(noisy_data)


            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        history["train_loss"].append(epoch_train_loss)


        # --- validation ---
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                noisy_data = add_gaussian_noise(data)

                output = model(noisy_data)
                if isinstance(output, tuple):
                    output = output[0]

                loss = criterion(output, data)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(test_loader)
        history["val_loss"].append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f}"
        )

        # сохраняем лучшую модель
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

    return history, best_model_path

def plot_history(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_reconstruction(best_model_path: str, data_dir: str = "./data"):
    device = get_device()

    # такая же конфигурация, как в train()
    model = AutoEncoder(
        in_channels=1,
        latent_dim=256,
        conv_num=2,
        shape=(28, 28),
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    _, test_loader = MNIST(path=data_dir, batch_size=64)

    # берём один батч
    data, _ = next(iter(test_loader))
    # случайный индекс в батче
    idx = random.randint(0, data.size(0) - 1)

    img = data[idx:idx + 1].to(device)  # [1, 1, 28, 28]
    noisy = add_gaussian_noise(img)

    with torch.no_grad():
        output = model(noisy)
        if isinstance(output, tuple):
            output = output[0]

    # переносим на CPU и убираем лишние размерности
    img_np = img.squeeze().cpu().numpy()
    noisy_np = noisy.squeeze().cpu().numpy()
    rec_np = output.squeeze().cpu().numpy()

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Noisy")
    plt.imshow(noisy_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    plt.imshow(rec_np, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    history, best_model_path = train()
    plot_history(history)
    visualize_reconstruction(best_model_path)