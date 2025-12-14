from torch import nn


class GAN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 784),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        x = x.view(x.size(0), -1)
        return self.discriminator(x)

    def forward(self, x, z):
        x = x.view(x.size(0), -1)
        generator = self.generate(z)
        real = self.discriminator(x)
        fake = self.discriminator(generator)

        return generator, real, fake
