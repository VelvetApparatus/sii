import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from db.mnist import MNIST
from models.gan import GAN_VAE, GAN
import torch
from torchvision.utils import save_image
import os
from pkg.device import get_device


def save_generated_samples(
        model,
        device,
        latent_dim,
        save_dir="./data/gen/gan_vae",
        n_samples=15,
):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        x_fake = model.generate(z)  # (n, 784)

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
        latent_dim=20,
        lr_g=2e-4,
        lr_d=2e-4,
):
    train_loader, test_loader = MNIST(path, batch_size)

    model = GAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    # bce_logits = nn.BCEWithLogitsLoss()

    loss = nn.BCELoss()

    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g)

    model.train()



    for epoch in tqdm(range(epochs), "training epochs"):
        d_loss_meter = 0.0
        g_loss_meter = 0.0
        n_batches = 0
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            # Discriminator train
            real_imgs = real_imgs.to(device)
            real_labels = model.discriminate(real_imgs)
            true_real_labels = torch.ones(real_imgs.shape[0], 1, device=device)

            # noise = torch.randn(real_imgs.size(0), latent_dim, device=device)
            noise = (torch.rand(real_imgs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)
            print(f"noise shape: {noise.shape}")

            fake_imgs = model.generate(noise)
            fake_label = model.discriminate(fake_imgs)
            true_fake_labels = torch.zeros(fake_imgs.shape[0], 1, device=device)

            outputs = torch.cat([real_labels, fake_label], dim=0)
            targets = torch.cat([true_real_labels, true_fake_labels], dim=0)

            d_loss = loss(outputs, targets)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # noise = torch.randn(real_imgs.size(0), latent_dim, device=device)
            noise = (torch.rand(real_imgs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            fake_imgs = model.generate(noise)


            # Generator
            noise = torch.randn(fake_imgs.size(0), latent_dim, device=device)
            fake_imgs = model.generate(noise)
            fake_labels = model.discriminate(fake_imgs)
            fake_targets = torch.zeros(fake_imgs.size(0), 1, device=device)

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

    # for epoch in tqdm(range(epochs), desc="Epochs"):
    #     d_loss_meter = 0.0
    #     g_loss_meter = 0.0
    #     vae_loss_meter = 0.0
    #     n_batches = 0
    #
    #     for batch_idx, (real_imgs, _) in enumerate(train_loader):
    #         real_imgs = real_imgs.to(device)
    #         B = real_imgs.size(0)
    #
    #         # flatten real images: (B, 784)
    #         x_real_flat = real_imgs.view(B, -1)
    #
    #         # =========================
    #         # 1) VAE step (recon + KL)
    #         # =========================
    #         x_hat, mu, logvar = model.vae(x_real_flat)
    #
    #         recon_loss = F.binary_cross_entropy(x_hat, x_real_flat, reduction="sum") / B
    #
    #         kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    #
    #         vae_loss = recon_loss + kl
    #
    #         opt_g.zero_grad()
    #         (vae_weight * vae_loss).backward()
    #         opt_g.step()
    #
    #         # =========================
    #         # 2) D step: real vs fake
    #         # =========================
    #         with torch.no_grad():
    #             noise = torch.randn(B, latent_dim, device=device)
    #             fake_imgs = model.generate(noise)  # (B,784)
    #
    #         d_real = model.discriminate(x_real_flat)  # logits
    #         d_fake = model.discriminate(fake_imgs)  # logits
    #
    #         d_loss = (
    #                 bce_logits(d_real, torch.ones_like(d_real)) +
    #                 bce_logits(d_fake, torch.zeros_like(d_fake))
    #         )
    #
    #         opt_d.zero_grad()
    #         d_loss.backward()
    #         opt_d.step()
    #
    #         # =========================
    #         # 3) G (decoder) adv step
    #         # =========================
    #         # Важно: обучаем генератор "обманывать" D.
    #         noise = torch.randn(B, latent_dim, device=device)
    #         fake_imgs = model.generate(noise)
    #         d_fake_for_g = model.discriminate(fake_imgs)
    #
    #         g_adv_loss = bce_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))
    #
    #         # Здесь мы оптимизируем VAE-параметры, но по факту adversarial градиент пойдёт в decoder.
    #         opt_g.zero_grad()
    #         (adv_weight * g_adv_loss).backward()
    #         opt_g.step()
    #
    #         # meters
    #         d_loss_meter += d_loss.item()
    #         g_loss_meter += g_adv_loss.item()
    #         vae_loss_meter += vae_loss.item()
    #         n_batches += 1
    #
    #     print(
    #         f"epoch {epoch + 1}/{epochs} | "
    #         f"vae_loss={vae_loss_meter / n_batches:.4f} | "
    #         f"d_loss={d_loss_meter / n_batches:.4f} | "
    #         f"g_adv_loss={g_loss_meter / n_batches:.4f}"
    #     )

    return model


def main(gen_model):
    device = get_device()
    batch_size = 100

    if gen_model == "vae":
        gen_model = train_gan_vae(
            device,
            batch_size,
            150,
            path="./data"
        )
        save_generated_samples(
            model=gen_model,
            device=device,
            latent_dim=20,
            save_dir="generated_after_training",
            n_samples=15,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main(gen_model="vae")
