import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> [32, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> [32, 64, 64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [64, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> [64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# -> [128, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> [128, 16, 16]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# -> [256, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> [256, 8, 8]

            nn.Flatten(),  # -> [batch, 256*8*8 = 16384]
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),                # -> [256, 8, 8]

            nn.ConvTranspose2d(256, 128, 2, stride=2),   # -> [128, 16, 16]
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),    # -> [64, 32, 32]
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),     # -> [32, 64, 64]
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 2, stride=2),      # -> [1, 128, 128]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
