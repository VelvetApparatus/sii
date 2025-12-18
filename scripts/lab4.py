import torch
from tqdm import tqdm

from db.mnist import MNIST
from models.diffusion import sinusoidal_embedding, DDPM_MLP
from pkg.device import get_device
from torchvision.utils import save_image


def get_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def sample(model, num_samples, device, timesteps=1000):
    model.eval()
    with torch.no_grad():
        samples = torch.randn(num_samples, 784, device=device)
        for t in reversed(range(timesteps)):
            t_emb = sinusoidal_embedding(torch.tensor([t] * num_samples, device=device), dim=128)
            samples = model(samples, t_emb)
        return samples


def train_epoch(model, dataloader, optimizer, device, timesteps=1000):
    model.train()
    pbar = tqdm(dataloader, desc="Training")
    for x, _ in pbar:
        x = x.view(x.size(0), -1).to(device)  # Flatten the images to 784-dimensional vectors

        # Random time steps
        t = torch.randint(0, timesteps, (x.size(0),), device=device)
        t_emb = sinusoidal_embedding(t, 128)

        # Forward pass
        optimizer.zero_grad()
        pred = model(x, t_emb)

        # Compute loss
        loss = get_loss(pred, x)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())



def train_model():
    device = get_device()
    train_loader, _ = MNIST()
    model = DDPM_MLP(input_dim=784, base_dim=512, t_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(model, train_loader, optimizer, device)

    # Sample from the model after training
    num_samples = 16
    samples = sample(model, num_samples, device)

    # Reshape and save images
    samples = samples.view(num_samples, 1, 28, 28)
    save_image(samples, './data/gen/diffusion/generated_samples.png')




if __name__ == "__main__":
    train_model()

