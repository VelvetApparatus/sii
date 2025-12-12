import torch.nn.functional as F
import torch

from db.mnist import MNIST
from pkg.device import get_device
from models.cvae import CVAE
from torchvision.utils import save_image
import os
from tqdm import tqdm


def cvae_loss(x_hat, x, mu, logvar):
    # x, x_hat: (B, 784)
    # реконструкция
    recon_loss = F.binary_cross_entropy(
        x_hat,
        x.view(x.size(0), -1),
        reduction="sum"
    )
    # KL дивергенция
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


def train(
        num_epochs: int = 100,
        out_dir: str = "./data/gen/cvae_outputs",
):
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    model = CVAE().to(device)

    train_loader, test_loader = MNIST()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(num_epochs), "training"):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)  # (B,1,28,28)
            y = y.to(device)  # (B,)

            x_hat, mu, logvar = model(x, y)
            loss = cvae_loss(x_hat, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

        # # --- Генерация и сохранение условных сэмплов на каждой эпохе (например, цифра 3) ---
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_cond = torch.full((16,), 3, dtype=torch.long, device=device)  # все тройки
                z = torch.randn(16, model.latent_dim, device=device)
                x_gen = model.decode(z, y_cond)  # (16, 784)
                x_gen = x_gen.view(-1, 1, 28, 28)  # (16, 1, 28, 28)

                save_image(
                    x_gen,
                    os.path.join(out_dir, f"epoch_{epoch+1:03d}_samples_digit3.png"),
                    nrow=4,
                    normalize=True,
                    value_range=(0, 1),
                )

    # --- После обучения: сохраняем реконструкции с тестового датасета ---
    model.eval()
    with torch.no_grad():
        x_test, y_test = next(iter(test_loader))
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        x_hat, mu, logvar = model(x_test, y_test)
        x_hat = x_hat.view(-1, 1, 28, 28)

        # возьмём первые 16 картинок
        x_orig_16 = x_test[:16]
        x_recon_16 = x_hat[:16]

        save_image(
            x_orig_16,
            os.path.join(out_dir, "test_original_first16.png"),
            nrow=4,
            normalize=True,
            value_range=(0, 1),
        )

        save_image(
            x_recon_16,
            os.path.join(out_dir, "test_reconstructed_first16.png"),
            nrow=4,
            normalize=True,
            value_range=(0, 1),
        )

    return model


if __name__ == "__main__":
    train()
