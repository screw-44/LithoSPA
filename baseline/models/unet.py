
import torch


from torch import nn

import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Unet, self).__init__()

        self.encoder1 = self.conv_stage(n_channels, 64)
        self.encoder2 = self.conv_stage(64, 128)
        self.encoder3 = self.conv_stage(128, 256)
        self.encoder4 = self.conv_stage(256, 512)

        self.bottleneck = self.conv_stage(512, 1024)

        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_stage(1024, 512)
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_stage(512, 256)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_stage(256, 128)
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_stage(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    @staticmethod
    def conv_stage(in_channels=1, out_channels=1):
        stage = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return stage

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        bottle = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        dec4 = self.up_conv4(bottle)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)

        return out