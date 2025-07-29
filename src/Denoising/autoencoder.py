import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder_gated(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_gated, self).__init__()
        self.gate_masks = []

        # Encoder layers with batch normalization
        self.encoder_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.encoder_bn1 = nn.BatchNorm2d(32)
        self.encoder_pool1 = nn.MaxPool2d(2, 2)

        self.encoder_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.encoder_bn2 = nn.BatchNorm2d(64)
        self.encoder_pool2 = nn.MaxPool2d(2, 2)

        self.encoder_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_bn3 = nn.BatchNorm2d(128)
        self.encoder_pool3 = nn.MaxPool2d(2, 2)

        self.encoder_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_bn4 = nn.BatchNorm2d(256)
        self.encoder_pool4 = nn.MaxPool2d(2, 2)

        self.encoder_conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_bn5 = nn.BatchNorm2d(512)
        self.encoder_pool5 = nn.MaxPool2d(2, 2)

        self.bottleneck_encoder = nn.Linear(4 * 4 * 512, 64)
        self.bottleneck_bn = nn.BatchNorm1d(64)
        self.bottleneck_dropout = nn.Dropout(0.3)

        self.bottleneck_decoder = nn.Linear(64, 4 * 4 * 512)

        self.decoder_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(256)

        self.decoder_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(128)

        self.decoder_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(64)

        self.decoder_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(32)

        self.decoder_conv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        # Gated skip connection layers
        self.gate3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )
        self.skip_conv2 = nn.Conv2d(384, 128, kernel_size=1)

        self.gate2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )
        self.skip_conv3 = nn.Conv2d(192, 64, kernel_size=1)

        self.gate1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.skip_conv4 = nn.Conv2d(96, 32, kernel_size=1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(0.2)

    def encode(self, x):
        skip_connections = []

        x = self.leaky_relu(self.encoder_bn1(self.encoder_conv1(x)))
        skip_connections.append(x)
        x = self.encoder_pool1(x)

        x = self.leaky_relu(self.encoder_bn2(self.encoder_conv2(x)))
        skip_connections.append(x)
        x = self.encoder_pool2(x)

        x = self.leaky_relu(self.encoder_bn3(self.encoder_conv3(x)))
        skip_connections.append(x)
        x = self.encoder_pool3(x)

        x = self.leaky_relu(self.encoder_bn4(self.encoder_conv4(x)))
        skip_connections.append(x)
        x = self.encoder_pool4(x)

        x = self.leaky_relu(self.encoder_bn5(self.encoder_conv5(x)))
        x = self.encoder_pool5(x)

        x = x.view(x.size(0), -1)
        encoded = self.leaky_relu(self.bottleneck_bn(self.bottleneck_encoder(x)))
        encoded = self.bottleneck_dropout(encoded)

        return encoded, skip_connections

    def decode(self, encoded, skip_connections):
        x = self.leaky_relu(self.bottleneck_decoder(encoded))
        x = x.view(x.size(0), 512, 4, 4)

        x = self.leaky_relu(self.decoder_bn1(self.decoder_conv1(x)))
        x = self.dropout(x)

        x = self.leaky_relu(self.decoder_bn2(self.decoder_conv2(x)))
        skip_3 = self.dropout(skip_connections[3])
        gate_mask3 = self.gate3(skip_3)
        self.gate_masks.append(gate_mask3)
        gated_3 = gate_mask3 * skip_3
        x = torch.cat([x, gated_3], dim=1)
        x = self.leaky_relu(self.skip_conv2(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.decoder_bn3(self.decoder_conv3(x)))
        skip_2 = self.dropout(skip_connections[2])
        gate_mask2 = self.gate2(skip_2)
        self.gate_masks.append(gate_mask2)
        gated_2 = gate_mask2 * skip_2
        x = torch.cat([x, gated_2], dim=1)
        x = self.leaky_relu(self.skip_conv3(x))
        x = self.dropout(x)

        x = self.leaky_relu(self.decoder_bn4(self.decoder_conv4(x)))
        skip_1 = self.dropout(skip_connections[1])
        gate_mask1 = self.gate1(skip_1)
        self.gate_masks.append(gate_mask1)
        gated_1 = gate_mask1 * skip_1
        x = torch.cat([x, gated_1], dim=1)
        x = self.leaky_relu(self.skip_conv4(x))

        x = self.decoder_conv5(x)
        reconstructed = torch.sigmoid(x)

        return reconstructed

    def forward(self, x):
        self.gate_masks = []
        encoded, skip_connections = self.encode(x)
        reconstructed = self.decode(encoded, skip_connections)
        return reconstructed, encoded, self.gate_masks

class DenoisingLoss(nn.Module):
    def __init__(self, recon_weight=1.0, smooth_weight=0, sparsity_weight=0, gates_weight = 0):
        super(DenoisingLoss, self).__init__()
        self.recon_weight = recon_weight    # legacy unused
        self.smooth_weight = smooth_weight  # legacy unused
        self.sparsity_weight = sparsity_weight  # legacy unused
        self.gates_weight = gates_weight
        self.mse = nn.MSELoss()                 # legacy unused
        self.L1 = nn.L1Loss()
        self.gates_loss = 0.0

    def forward(self, reconstructed, target, encoded=None, gates=None):
        # Main reconstruction loss
        recon_loss = self.L1(reconstructed, target)

        # Smoothness loss - penalize high-frequency noise - legacy unused
        # Laplacian kernel for detecting rapid changes    - legacy unused
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                      dtype=torch.float32, device=reconstructed.device).unsqueeze(0).unsqueeze(0)

        laplacian_recon = F.conv2d(reconstructed, laplacian_kernel, padding=1)
        smoothness_loss = torch.mean(torch.abs(laplacian_recon))

        # Sparsity loss on encoded features (L1 regularization) - legacy unused
        sparsity_loss = 0
        if encoded is not None:
            sparsity_loss = torch.mean(torch.abs(encoded))

        self.gates_loss = 0.0
        if gates is not None:
            for mask in gates:
                self.gates_loss += mask.mean()

        total_loss = (self.recon_weight * recon_loss +
                     self.smooth_weight * smoothness_loss +
                     self.sparsity_weight * sparsity_loss +
                      self.gates_weight * self.gates_loss
                      )

        return total_loss

