import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,   # 28x28
        hidden_dim: int = 400,
        latent_dim: int = 20,
        num_classes: int = 10,  # MNIST
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # ----- Encoder -----
        # На вход энкодеру даём [x_flat, y_one_hot]
        enc_input_dim = input_dim + num_classes

        self.fc1 = nn.Linear(enc_input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ----- Decoder -----
        # На вход декодеру даём [z, y_one_hot]
        dec_input_dim = latent_dim + num_classes

        self.fc2 = nn.Linear(dec_input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        """
        x: (B, 784) или (B, 1, 28, 28)
        y: (B,) - целочисленные метки классов 0..9
        """
        if x.dim() == 4:
            # (B, 1, 28, 28) -> (B, 784)
            x = x.view(x.size(0), -1)

        # y -> one-hot: (B, num_classes)
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()

        # конкатенируем по признакам: (B, 784+10)
        xy = torch.cat([x, y_onehot], dim=1)

        h = F.relu(self.fc1(xy))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        """
        z: (B, latent_dim)
        y: (B,) - те же классы, что и при encode
        """
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        zy = torch.cat([z, y_onehot], dim=1)  # (B, latent_dim + num_classes)

        h = F.relu(self.fc2(zy))
        x_hat = torch.sigmoid(self.fc3(h))     # (B, 784)
        return x_hat

    def forward(self, x, y):
        """
        x: (B, 1, 28, 28) или (B, 784)
        y: (B,)
        """
        if x.dim() == 4:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x

        mu, logvar = self.encode(x_flat, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar
