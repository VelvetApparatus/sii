import os
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from db.mnist import MNIST
from models.autoregression import PixelCNN
import torch

from pkg.device import get_device
from pkg.history_writer import HistoryWriter


N_BINS = 16
def quantize(x):
    # x float [0,1]
    return (x * (N_BINS - 1)).round().clamp(0, N_BINS-1).long()


def to_uint8(x):
    # x: (B,1,28,28) float in [0,1]
    xq = (x * 255.0).round().clamp(0, 255).long()
    return xq


def pixelcnn_loss(model, x_float):
    # x_float: (B,1,28,28) float in [0,1]
    x_q = quantize(x_float)      # <-- ВОТ ЗДЕСЬ

    logits = model(x_q)          # (B, N_BINS, H, W)
    target = x_q[:, 0]           # (B, H, W)

    loss = F.cross_entropy(logits, target, reduction="mean")
    return loss



def train_pixelcnn(
        epochs=20,
        out_dir="./data/gen/pixelcnn_out",
):
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    history = HistoryWriter()

    train_loader, _ = MNIST()

    model = PixelCNN(
        hidden=256,
        n_layers=16,
        n_embeddings=N_BINS,
    ).to(device)


    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
     

        pbar = tqdm(train_loader, desc=f"epoch {ep}")
        for x, _ in pbar:
            x = x.to(device)
            loss = pixelcnn_loss(model, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += loss.item() * bs
            seen += bs

            pbar.set_postfix(avg_loss=total / seen, batch_loss=loss.item())
        sched.step()

        avg_loss = total / seen
        print(f"Epoch {ep}: avg_loss={avg_loss:.4f}")
        history.save_record("loss", avg_loss)

        if ep % 5 == 0:
            xs = model.sample(batch_size=16, device=device)   # long 0..N_BINS-1
            xs = (xs.float() / (N_BINS - 1)).clamp(0, 1)
            save_image(xs, os.path.join(out_dir, f"samples_ep{ep:03d}.png"), nrow=4)



    history.plot_history("epochs", "./data/plots/autoencoder.png")
    return model


if __name__ == "__main__":
    train_pixelcnn(40)