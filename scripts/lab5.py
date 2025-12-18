import os
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda
from tqdm import tqdm
from torchvision.utils import save_image

from db.mnist import MNIST
from models.autoregression import PixelCNN
import torch

from pkg.device import get_device


def to_uint8(x):
    # x: (B,1,28,28) float in [0,1]
    xq = (x * 255.0).round().clamp(0, 255).long()
    return xq


def pixelcnn_loss(model, x_float):
    # x = to_uint8(x_float)
    logits = model(x)  # (B,256,H,W)
    # target: (B,H,W)
    target = x[:, 0]
    loss = F.cross_entropy(logits, target, reduction="mean")
    return loss


def train_pixelcnn(
        epochs=20,
        out_dir="./data/gen/pixelcnn_out",
):
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    t = Compose([
        ToTensor(),
        Lambda(lambda x: to_uint8(x)),
    ])
    train_loader, _ = MNIST(transform=t)

    model = PixelCNN(
        hidden=64,
        n_layers=7,
        n_embeddings=256,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0

        for x, _ in tqdm(train_loader, desc=f"epoch {ep}"):
            x = x.to(device)
            loss = pixelcnn_loss(model, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * x.size(0)

        print(f"Epoch {ep}: loss={total / len(train_loader.dataset):.4f}")

        if ep % 5 == 0:
            xs = model.sample(batch_size=16, device=device)  # long 0..255
            xs = (xs.float() / 255.0).clamp(0, 1)
            save_image(xs, os.path.join(out_dir, f"samples_ep{ep:03d}.png"), nrow=4)

    return model
