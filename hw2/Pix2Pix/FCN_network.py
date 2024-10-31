import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
            super().__init__()

            # Encoder (Convolutional Layers)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            # Decoder (Deconvolutional Layers)
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            )
            self.deconv4 = nn.Sequential(
                nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # Output channels: 3 for RGB
                nn.Sigmoid()  # Sigmoid activation for output in range [0, 1]
            )
    def forward(self, x):
        # Encoder forward pass
        e1 = self.conv1(x)  # output size: 8 channels
        e2 = self.conv2(e1)  # output size: 16 channels
        e3 = self.conv3(e2)  # output size: 32 channels
        e4 = self.conv4(e3)  # output size: 64 channels

        # Decoder forward pass
        d1 = self.deconv1(e4) + e3  # Skip connection
        d2 = self.deconv2(d1) + e2  # Skip connection
        d3 = self.deconv3(d2) + e1  # Skip connection
        output = self.deconv4(d3)  # Final layer
        
        return output
    