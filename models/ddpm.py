import math
import os
import torch
from torch import nn
from torch import amp
from tqdm import tqdm

# (опционально) для сохранения картинок
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image


# -------------------------
# utils
# -------------------------
def get(v: torch.Tensor, t: torch.Tensor, x_shape=None):
    """
    v: Tensor [T]
    t: LongTensor [B]
    return: Tensor [B, 1, 1, 1] (или под x_shape)
    """
    out = v.gather(0, t)  # [B]
    if x_shape is None:
        return out.view(-1, 1, 1, 1)
    # подгоняем под размерность x (B, C, H, W) или (B, ...)
    return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))


class MeanMetric:
    """Простая замена torchmetrics.MeanMetric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += int(n)

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.sum / self.count)


def default_inverse_transform(x):
    """
    Если у тебя нормализация была [-1, 1], то это обратное преобразование в [0, 255].
    Подстрой под свой пайплайн датасета.
    """
    x = (x.clamp(-1, 1) + 1) / 2  # -> [0,1]
    x = (x * 255.0).round().clamp(0, 255)
    return x


# -------------------------
# Diffusion
# -------------------------
class DiffusionModel(nn.Module):
    def __init__(self, diffusion_timestamps=1000, img_shape=(1, 28, 28), device="cpu"):
        super().__init__()
        self.diffusion_timestamps = diffusion_timestamps
        self.img_shape = img_shape
        self.device = device

        self.init()  # важно: сразу считаем расписания

    def init(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()                               # [T]
        self.alpha = 1.0 - self.beta                               # [T]

        self.sqrt_beta = torch.sqrt(self.beta)                     # [T]
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)   # [T]
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1.0 - self.alpha_cumulative)

    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        # у тебя было self.num_diffusion_timesteps -> должно быть self.diffusion_timestamps
        scale = 1000 / self.diffusion_timestamps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.diffusion_timestamps,
            dtype=torch.float32,
            device=self.device,
        )

    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        """
        q(x_t|x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        """
        eps = torch.randn_like(x0)
        mean = get(self.sqrt_alpha_cumulative, t=timesteps, x_shape=x0.shape) * x0
        std = get(self.sqrt_one_minus_alpha_cumulative, t=timesteps, x_shape=x0.shape)
        xt = mean + std * eps
        return xt, eps


# -------------------------
# train
# -------------------------
def train_one_epoch(
    model,
    loader,
    sd: DiffusionModel,
    optimizer,
    scaler,
    loss_fn,
    device,
    epoch=1,
    num_epochs=1,
    timesteps=1000,
):
    loss_record = MeanMetric()
    model.train()

    # amp.autocast: корректно через device_type
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{num_epochs}")

        for x0s, _ in loader:
            tq.update(1)

            x0s = x0s.to(device)
            # t обычно берут [0..T-1] или [1..T-1]; ниже оставим [1..T-1], как у тебя
            ts = torch.randint(low=1, high=timesteps, size=(x0s.shape[0],), device=device)

            xts, gt_noise = sd.forward_diffusion(x0s, ts)

            with amp.autocast(device_type=device_type):
                pred_noise = model(xts, ts)
                loss = loss_fn(pred_noise, gt_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().item())
            loss_record.update(loss_value, n=x0s.shape[0])
            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = float(loss_record.compute().item())
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss


# -------------------------
# sampling (reverse diffusion)
# -------------------------
@torch.no_grad()
def reverse_diffusion(
    model,
    sd: DiffusionModel,
    timesteps=1000,
    img_shape=(1, 28, 28),
    num_images=16,
    nrow=8,
    device="cpu",
    save_path="sample.png",
    inverse_transform=default_inverse_transform,
):
    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    for time_step in tqdm(
        iterable=reversed(range(1, timesteps)),
        total=timesteps - 1,
        dynamic_ncols=True,
        desc="Sampling :: ",
        position=0,
    ):
        ts = torch.full((num_images,), time_step, dtype=torch.long, device=device)
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t = get(sd.beta, ts, x_shape=x.shape)
        one_by_sqrt_alpha_t = get(sd.one_by_sqrt_alpha, ts, x_shape=x.shape)
        sqrt_one_minus_alpha_cum_t = get(sd.sqrt_one_minus_alpha_cumulative, ts, x_shape=x.shape)

        x = one_by_sqrt_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cum_t) * predicted_noise) + torch.sqrt(beta_t) * z

    # save result
    x_img = inverse_transform(x).to(torch.uint8)
    grid = make_grid(x_img, nrow=nrow, pad_value=255.0)
    pil_image = TF.to_pil_image(grid.cpu())

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    pil_image.save(save_path)
    return pil_image


# -------------------------
# example scaler
# -------------------------
def make_grad_scaler(device):
    # Для CPU scaler обычно не нужен, но пусть будет единообразно
    return amp.GradScaler(enabled=str(device).startswith("cuda"))
