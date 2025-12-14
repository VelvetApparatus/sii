import torch
from tqdm import tqdm
from torchvision.utils import save_image

from db.mnist import MNIST
from models.diffusion import sinusoidal_embedding, DDPM_MLP
from pkg.device import get_device


# -----------------------
# DDPM schedule
# -----------------------
def make_beta_schedule(timesteps, device, beta_start=1e-4, beta_end=2e-2):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t
    return betas, alphas, alpha_bars


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """
    a: [T], t: [B] (long), return: [B, 1] (или [B,1,1,1] если нужно)
    """
    out = a.gather(0, t)
    return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))


# -----------------------
# forward diffusion q(x_t|x0)
# -----------------------
def q_sample(x0, t, alpha_bars):
    """
    x0: [B, 784] in [-1,1]
    returns: x_t, noise eps
    """
    eps = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(extract(alpha_bars, t, x0.shape))
    sqrt_1mab = torch.sqrt(1.0 - extract(alpha_bars, t, x0.shape))
    xt = sqrt_ab * x0 + sqrt_1mab * eps
    return xt, eps


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


# -----------------------
# training epoch
# -----------------------
def train_epoch(model, dataloader, optimizer, device, betas, alphas, alpha_bars):
    model.train()
    timesteps = betas.shape[0]
    pbar = tqdm(dataloader, desc="Training", dynamic_ncols=True)

    for x, _ in pbar:
        # x: [B,1,28,28] -> [B,784]
        x0 = x.view(x.size(0), -1).to(device)

        # Важно: если MNIST() уже нормализует в [-1,1], оставь как есть.
        # Если x0 в [0,1], то лучше привести к [-1,1]:
        # x0 = x0 * 2 - 1

        t = torch.randint(0, timesteps, (x0.size(0),), device=device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t, dim=128)

        # forward diffusion
        xt, eps = q_sample(x0, t, alpha_bars)

        optimizer.zero_grad()
        eps_pred = model(xt, t_emb)          # модель предсказывает шум
        loss = mse(eps_pred, eps)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=float(loss.item()))


# -----------------------
# sampling (reverse)
# -----------------------
@torch.no_grad()
def sample(model, num_samples, device, betas, alphas, alpha_bars):
    model.eval()
    T = betas.shape[0]

    x = torch.randn(num_samples, 784, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        t_emb = sinusoidal_embedding(t_batch, dim=128)

        eps_pred = model(x, t_emb)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        # коэффициенты
        one_over_sqrt_alpha = 1.0 / torch.sqrt(alpha_t)
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)

        # DDPM mean
        mean = one_over_sqrt_alpha * (x - (beta_t / sqrt_one_minus_ab) * eps_pred)

        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean

    return x


# -----------------------
# full train loop
# -----------------------
def train_model():
    device = get_device()
    train_loader, _ = MNIST()

    model = DDPM_MLP(input_dim=784, base_dim=512, t_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    timesteps = 1000
    betas, alphas, alpha_bars = make_beta_schedule(timesteps, device=device)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(model, train_loader, optimizer, device, betas, alphas, alpha_bars)

        # (опционально) сэмпл после каждой эпохи
        samples = sample(model, num_samples=16, device=device,
                         betas=betas, alphas=alphas, alpha_bars=alpha_bars)
        samples = samples.view(16, 1, 28, 28)

        # если обучали на [-1,1], то обратно в [0,1]
        samples = (samples.clamp(-1, 1) + 1) / 2
        if epoch % 10 == 0:
            save_image(samples, f'./data/gen/diffusion/samples_epoch_{epoch+1}.png', nrow=4)

    # финальный сэмпл
    samples = sample(model, num_samples=16, device=device,
                     betas=betas, alphas=alphas, alpha_bars=alpha_bars)
    samples = samples.view(16, 1, 28, 28)
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, './data/gen/diffusion/generated_samples.png', nrow=4)


if __name__ == "__main__":
    train_model()
