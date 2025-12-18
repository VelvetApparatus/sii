import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Masked conv для автогрессии.
    mask_type 'A' — запрещает смотреть на текущий пиксель (первый слой)
    mask_type 'B' — разрешает смотреть на текущий пиксель (последующие)
    """
    def __init__(self, mask_type, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=True)
        assert mask_type in ["A", "B"]
        self.register_buffer("mask", torch.ones_like(self.weight))

        kH, kW = self.weight.shape[-2], self.weight.shape[-1]
        yc, xc = kH // 2, kW // 2

        # всё "будущее" (ниже текущей строки) запрещаем
        self.mask[:, :, yc + 1 :, :] = 0
        # в текущей строке запрещаем пиксели справа от центра
        self.mask[:, :, yc, xc + 1 :] = 0
        # для типа A запрещаем и центр (текущий пиксель)
        if mask_type == "A":
            self.mask[:, :, yc, xc] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, n_embeddings=256, hidden=64, n_layers=7):
        super().__init__()
        self.n_embeddings = n_embeddings

        # вход: (B,1,H,W) значения 0..255 (long) -> one-hot (B,256,H,W)
        self.conv_in = MaskedConv2d("A", n_embeddings, hidden, kernel_size=7, padding=3)

        blocks = []
        for _ in range(n_layers):
            blocks += [
                MaskedConv2d("B", hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
        self.net = nn.Sequential(*blocks)

        # выход: logits для каждого уровня интенсивности (256)
        self.conv_out = nn.Conv2d(hidden, n_embeddings, kernel_size=1)

    def forward(self, x_long):
        """
        x_long: (B,1,H,W) long in [0..255]
        returns logits: (B,256,H,W)
        """
        B, C, H, W = x_long.shape
        assert C == 1
        x_oh = F.one_hot(x_long[:, 0], num_classes=self.n_embeddings).permute(0, 3, 1, 2).float()
        h = F.relu(self.conv_in(x_oh))
        h = self.net(h)
        logits = self.conv_out(h)
        return logits

    @torch.no_grad()
    def sample(self, batch_size=16, H=28, W=28, device="cpu"):
        self.eval()
        x = torch.zeros((batch_size, 1, H, W), dtype=torch.long, device=device)

        for i in range(H):
            for j in range(W):
                logits = self.forward(x)          # (B,256,H,W)
                probs = F.softmax(logits[:, :, i, j], dim=1)  # (B,256)
                x[:, 0, i, j] = torch.multinomial(probs, num_samples=1).squeeze(1)

        return x  # long 0..255
