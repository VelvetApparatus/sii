from torch import nn
from tqdm import tqdm
from db.mnist import MNIST
from models.gan import GAN
import torch
from torchvision.utils import save_image
import os
from pkg.device import get_device


def save_generated_samples(
        model,
        device,
        save_dir="./data/gen/gan_vae",
        n_samples=15,
):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        noise = (torch.rand(n_samples, 128) - 0.5) / 0.5
        noise = noise.to(device)
        x_fake = model.generate(noise)  # (n, 784)
        x_fake = x_fake.clamp(0, 1)

        # reshape -> (n, 1, 28, 28)
        x_fake = x_fake.view(n_samples, 1, 28, 28)

        for i in range(n_samples):
            save_image(
                x_fake[i],
                os.path.join(save_dir, f"sample_{i}.png")
            )

    model.train()


def train_gan_vae(
        device,
        batch_size,
        epochs,
        path,
        input_dim=784,
        hidden_dim=400,
        lr_g=2e-4,
        lr_d=2e-4,
):
    train_loader, test_loader = MNIST(path, batch_size)

    model = GAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    loss = nn.BCEWithLogitsLoss()

    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g)

    model.train()

    for epoch in tqdm(range(epochs), "training epochs"):
        d_loss_meter = 0.0
        g_loss_meter = 0.0
        n_batches = 0
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            # Discriminator train

            # X
            real_imgs = real_imgs.to(device)
            # D(X)
            real_labels = model.discriminate(real_imgs)
            # Lt [true targets]
            true_real_labels = torch.ones(real_imgs.shape[0], 1, device=device)

            # Z
            noise = (torch.rand(real_imgs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            # G(Z)
            fake_imgs = model.generate(noise).detach()
            # D(G(Z))
            fake_label = model.discriminate(fake_imgs)
            # Lt [true targets]
            true_fake_labels = torch.zeros(fake_imgs.shape[0], 1, device=device)

            d_loss_real = loss(real_labels, true_real_labels)
            d_loss_fake = loss(fake_label, true_fake_labels)
            d_loss = d_loss_real + d_loss_fake

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # Generator
            # Z
            noise = (torch.rand(real_imgs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            # G(z)
            fake_imgs = model.generate(noise)
            # D(G(z))
            fake_labels = model.discriminate(fake_imgs)
            # Lt [true targets]
            fake_targets = torch.ones(fake_imgs.size(0), 1, device=device)


            g_loss = loss(fake_labels, fake_targets)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            # metrics
            d_loss_meter += d_loss.item()
            g_loss_meter += g_loss.item()
            n_batches += 1
        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"d_loss={d_loss_meter / n_batches:.4f} | "
            f"g_adv_loss={g_loss_meter / n_batches:.4f}"
        )

    return model


def main(gen_model):
    device = get_device()
    batch_size = 100

    if gen_model == "vae":
        gen_model = train_gan_vae(
            device,
            batch_size,
            150,
            path="./data/ds"
        )
        save_generated_samples(
            model=gen_model,
            device=device,
            save_dir="data/gen/gan",
            n_samples=15,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(gen_model="vae")
