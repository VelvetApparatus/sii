import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from db.mnist import MNIST
from pkg.device import get_device
from models.ddpm_diy import UNetDDPMDIY
from pkg.history_writer import HistoryWriter



# ---------- DDPM scheduler ----------
class DDPMScheduler:
    def __init__(
            self,
            T: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 2e-2,
            device="cpu",
    ):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)  # (T,)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

        # для sampling (обратный шаг)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_var = betas * (1.0 - alpha_bars.roll(1, 0)) / (1.0 - alpha_bars)
        self.posterior_var[0] = 0.0  # на t=0 шум не добавляем

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        a: (T,)
        t: (B,)
        returns: (B,1,1,1...) под x_shape
        """
        out = a.gather(0, t)  # (B,)
        return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        """
        s1 = self.extract(self.sqrt_alpha_bars, t, x0.shape)
        s2 = self.extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor):
        """
        Один обратный шаг: x_t -> x_{t-1}
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_ab_t = self.extract(self.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # предсказываем шум
        eps_hat = model(x, t)

        # DDPM mean: μ = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-a_bar_t) * eps_hat)
        model_mean = sqrt_recip_alpha_t * (x - (betas_t / sqrt_one_minus_ab_t) * eps_hat)

        # variance
        var_t = self.extract(self.posterior_var, t, x.shape)

        # если t==0, не добавляем шум
        noise = torch.randn_like(x)
        nonzero = (t != 0).float().view((t.shape[0],) + (1,) * (len(x.shape) - 1))
        return model_mean + nonzero * torch.sqrt(var_t) * noise

    @torch.no_grad()
    def sample(self, model, shape, device):
        """
        Полный sampling: x_T ~ N(0, I) -> ... -> x_0
        """
        model.eval()
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x


# ---------- training loss ----------
def ddpm_loss(model, scheduler: DDPMScheduler, x0: torch.Tensor):
    """
    x0: (B,1,28,28) in [0,1]
    """
    B = x0.size(0)
    device = x0.device
    t = torch.randint(0, scheduler.T, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = scheduler.q_sample(x0, t, noise)
    noise_hat = model(x_t, t)
    return F.mse_loss(noise_hat, noise, reduction="mean")


# ---------- main train loop ----------
def train(
    num_epochs: int = 50,
    out_dir: str = "./data/gen/ddpm_diy_outputs",
    T: int = 400,
):
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()

    # MNIST: 1 канал
    model = UNetDDPMDIY(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_mults=(1, 2,4),
        time_emb_dim=128,
    ).to(device)

    train_loader, test_loader = MNIST()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    scheduler = DDPMScheduler(T=T, device=device)
    pbar = tqdm(range(num_epochs), "training")

    history =HistoryWriter()


    for epoch in pbar:
        model.train()
        total_loss = 0.0
        idx = 0
        total = len(train_loader)
        for x, y in train_loader:
            idx += 1
            pbar.set_description(f"batch {idx}/{total}")
            x = x.to(device)  # (B,1,28,28) обычно уже [0,1]

            loss = ddpm_loss(model, scheduler, x)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        history.save_record("loss", avg_loss)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.6f}")

        # --- sampling каждые 10 эпох ---
        if epoch % 10 == 0:
            with torch.no_grad():
                x_gen = scheduler.sample(model, shape=(16, 1, 28, 28), device=device)
                # часто модель генерит примерно в [-1,1] или около того; тут ограничим для сохранения
                x_gen = torch.clamp(x_gen, 0.0, 1.0)

                save_image(
                    x_gen,
                    os.path.join(out_dir, f"epoch_{epoch+1:03d}_samples.png"),
                    nrow=4,
                    normalize=True,
                    value_range=(0, 1),
                )

    # --- после обучения: можно сохранить несколько сэмплов (или денойзить фиксированный шум) ---
    model.eval()
    with torch.no_grad():
        x_gen = scheduler.sample(model, shape=(64, 1, 28, 28), device=device)
        x_gen = torch.clamp(x_gen, 0.0, 1.0)
        save_image(
            x_gen,
            os.path.join(out_dir, "final_samples_64.png"),
            nrow=8,
            normalize=True,
            value_range=(0, 1),
        )



    history.plot_history("epochs", "./data/plots/ddpm_diy_history.png")

    return model


if __name__ == "__main__":
    train()
