from torch import nn
from torchvision import transforms
from db.mnist import MNIST
from models.gan import GAN
import torch
from torchvision.utils import save_image
import os
from pkg.device import get_device
import time
from torch.utils.tensorboard import SummaryWriter




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
        x_fake = x_fake * 0.5 + 0.5
        x_fake = x_fake.clamp(0, 1)

        # reshape -> (n, 1, 28, 28)
        x_fake = x_fake.view(n_samples, 1, 28, 28)

        for i in range(n_samples):
            save_image(
                x_fake[i],
                os.path.join(save_dir, f"sample_{i}.png")
            )

    model.train()



def train_gan(
        device,
        batch_size,
        epochs,
        path,
        input_dim=784,
        hidden_dim=400,
        lr_g=2e-4,
        # lr_d=5e-5,
        lr_d=2e-4,
        log_dir="runs/gan_mnist",
        checkpoint_dir="checkpoints/gan_mnist",
        update_d_every=1,          # <-- 1,2,5...
        save_every_epochs=10,      # периодические чекпойнты
        best_metric="p_fake",      # "p_fake" или "g_loss"
        warmup_epochs=3,           # не сохранять best в первые эпохи
):


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader, _ = MNIST(path, batch_size, transform=transform)

    model = GAN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    loss_fn = nn.BCELoss()

    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best.pt")

    # best tracking
    if best_metric == "p_fake":
        best_value = -1.0  # хотим максимизировать
    elif best_metric == "g_loss":
        best_value = float("inf")  # хотим минимизировать
    else:
        raise ValueError('best_metric must be "p_fake" or "g_loss"')

    global_step = 0
    model.train()

    for epoch in range(epochs):
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        d_steps = 0
        n_batches = 0

        d_real_prob_sum = 0.0
        d_fake_prob_sum = 0.0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            # --------- Discriminator update (каждый update_d_every батч) ---------
            d_loss = None
            if update_d_every > 0 and (batch_idx % update_d_every == 0):
                real_logits = model.discriminate(real_imgs)
                true_real = torch.full((bs, 1), 0.9, device=device)

                noise = (torch.rand(bs, 128, device=device) - 0.5) / 0.5
                fake_imgs = model.generate(noise).detach()
                fake_logits = model.discriminate(fake_imgs)
                true_fake = torch.zeros(bs, 1, device=device)

                d_loss_real = loss_fn(real_logits, true_real)
                d_loss_fake = loss_fn(fake_logits, true_fake)
                d_loss = d_loss_real + d_loss_fake

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                writer.add_scalar("loss/D_batch", d_loss.item(), global_step)
                writer.add_scalar("loss/D_real_batch", d_loss_real.item(), global_step)
                writer.add_scalar("loss/D_fake_batch", d_loss_fake.item(), global_step)

                d_loss_sum += d_loss.item()
                d_steps += 1

            # --------- Generator update ---------
            noise = (torch.rand(bs, 128, device=device) - 0.5) / 0.5
            fake_imgs = model.generate(noise)
            fake_logits = model.discriminate(fake_imgs)
            fake_targets = torch.ones(bs, 1, device=device)

            g_loss = loss_fn(fake_logits, fake_targets)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            writer.add_scalar("loss/G_batch", g_loss.item(), global_step)

            # Метрики вероятностей (через sigmoid(logits))
            with torch.no_grad():
                real_prob = torch.sigmoid(model.discriminate(real_imgs)).mean().item()
                fake_prob = torch.sigmoid(fake_logits).mean().item()
                writer.add_scalar("prob/D_real_batch", real_prob, global_step)
                writer.add_scalar("prob/D_fake_batch", fake_prob, global_step)

                d_real_prob_sum += real_prob
                d_fake_prob_sum += fake_prob

            g_loss_sum += g_loss.item()
            n_batches += 1
            global_step += 1

        # --------- epoch summaries ---------
        d_epoch = d_loss_sum / max(1, d_steps)
        g_epoch = g_loss_sum / max(1, n_batches)

        d_real_prob_epoch = d_real_prob_sum / max(1, n_batches)
        d_fake_prob_epoch = d_fake_prob_sum / max(1, n_batches)

        writer.add_scalar("loss/D_epoch", d_epoch, epoch)
        writer.add_scalar("loss/G_epoch", g_epoch, epoch)
        writer.add_scalar("prob/D_real_epoch", d_real_prob_epoch, epoch)
        writer.add_scalar("prob/D_fake_epoch", d_fake_prob_epoch, epoch)

        print(
            f"epoch {epoch+1}/{epochs} | "
            f"d_loss={d_epoch:.4f} | g_loss={g_epoch:.4f} | "
            f"D(real)={d_real_prob_epoch:.3f} | D(fake)={d_fake_prob_epoch:.3f}"
        )


        if (epoch + 1) % save_every_epochs == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt_d": opt_d.state_dict(),
                "opt_g": opt_g.state_dict(),
            }, ckpt_path)

        # --------- best checkpoint (after warmup) ---------
        if (epoch + 1) > warmup_epochs:
            if best_metric == "p_fake":
                current = d_fake_prob_epoch         # хотим, чтобы D принимал fake за real => prob выше
                improved = current > best_value
            else:  # "g_loss"
                current = g_epoch
                improved = current < best_value

            if improved:
                best_value = current
                torch.save({
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "best_value": best_value,
                    "model": model.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "opt_g": opt_g.state_dict(),
                }, best_path)
                writer.add_text("checkpoints/best", f"Saved best at epoch={epoch+1}, value={best_value}", epoch)

    writer.close()
    return model


def main():
    device = get_device()

    gen_model = train_gan(
        device=device,
        batch_size=64,
        epochs=50,
        path="./data/ds",
        update_d_every=2,
        best_metric="p_fake",
        checkpoint_dir="tensorboard/checkpoints/gan",
        log_dir="tensorboard/runs/gan",
    )

    save_generated_samples(
        model=gen_model,
        device=device,
        save_dir="data/gen/gan",
        n_samples=15,
    )



if __name__ == "__main__":
    main()
