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

        if conv_num <= 0:
            raise ValueError("conv_num должен быть >= 1")

        if isinstance(shape, int):
            h, w = shape, shape
        else:
            h, w = shape

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.conv_num = conv_num
        self.height = h
        self.width = w
        self.input_dim = in_channels * h * w

        # Глубина линейных блоков соответствует conv_num
        base_features = [512, 256, 128, 64, 32]
        if conv_num > len(base_features):
            raise ValueError(
                f"conv_num={conv_num} слишком большой, максимум {len(base_features)} "
                f"для текущего списка линейных слоёв."
            )

        hidden_dims = base_features[:conv_num]
        self.hidden_dims = hidden_dims

        # --- Encoder из линейных слоёв ---
        encoder_layers = []
        in_dim = self.input_dim
        for out_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder из линейных слоёв ---
        decoder_layers = []
        in_dim = latent_dim
        for out_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, self.input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)  # [B, input_dim]
        return self.encoder(x)             # [B, latent_dim]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)                # [B, input_dim]
        x = x.view(-1, self.in_channels, self.height, self.width)
        return x

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
