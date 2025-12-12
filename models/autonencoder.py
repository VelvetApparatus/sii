from torch import nn
import torch

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        conv_num: int = 4,
        shape: int | tuple[int, int] = 128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.conv_num = conv_num

        # --- Разбор shape ---
        if isinstance(shape, int):
            h, w = shape, shape
        else:
            h, w = shape

        if conv_num <= 0:
            raise ValueError("conv_num должен быть >= 1")

        # --- Проверка, что после conv_num пуллингов не уйдём в 0 ---
        h_enc, w_enc = h, w
        for i in range(conv_num):
            if h_enc < 2 or w_enc < 2:
                raise ValueError(
                    f"Слишком маленький shape={shape} для conv_num={conv_num}: "
                    f"на шаге {i + 1} MaxPool2d(2) приведёт к нулевому размеру."
                )
            h_enc //= 2
            w_enc //= 2

        if h_enc == 0 or w_enc == 0:
            raise ValueError(
                f"После {conv_num} пуллингов размер стал 0: "
                f"(h={h_enc}, w={w_enc}). Уменьши conv_num или увеличь shape."
            )

        self.h_enc = h_enc
        self.w_enc = w_enc

        # --- Каналы свёрток ---
        # Можно расширить этот список, если хочется более глубокие сети
        base_channels = [32, 64, 128, 256, 512]
        if conv_num > len(base_channels):
            raise ValueError(
                f"conv_num={conv_num} слишком большой, максимум {len(base_channels)} "
                f"для текущего списка каналов."
            )

        channels = base_channels[:conv_num]
        self.channels = channels  # [32, 64, 128, ...] длины conv_num

        # --- Encoder ---
        enc_layers = []
        in_ch = in_channels
        for out_ch in channels:
            enc_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            enc_layers.append(nn.ReLU(inplace=True))
            enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch

        self.encoder = nn.Sequential(*enc_layers)

        # --- Линейные слои для латент-вектора ---
        self.flat_dim = channels[-1] * self.h_enc * self.w_enc
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # --- Decoder ---
        dec_layers = []
        # каналов столько же, но идём в обратном порядке
        rev_channels = channels[::-1]

        in_ch = rev_channels[0]  # последний канал энкодера
        for out_ch in rev_channels[1:]:
            dec_layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=2,
                    stride=2,  # увеличиваем размер в 2 раза
                )
            )
            dec_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        # последний слой: из последнего числа каналов обратно в in_channels
        dec_layers.append(
            nn.ConvTranspose2d(
                in_ch,
                in_channels,
                kernel_size=2,
                stride=2,
            )
        )
        dec_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # [B, flat_dim]
        z = self.fc_enc(x)                 # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_dec(z)                 # [B, flat_dim]
        x = x.view(-1, self.channels[-1], self.h_enc, self.w_enc)
        x = self.decoder(x)                # [B, in_channels, H, W]
        return x

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
