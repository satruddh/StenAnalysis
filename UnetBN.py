import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.reduction = reduction
        self.fc_avg = nn.Linear(in_channels, in_channels // self.reduction)
        self.fc_out_avg = nn.Linear(in_channels // self.reduction, in_channels)

        self.fc_max = nn.Linear(in_channels, in_channels // self.reduction)
        self.fc_out_max = nn.Linear(in_channels // self.reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        
        channel_avg = torch.mean(x, dim=(2, 3), keepdim=True)
        channel_max = torch.max(x, dim=2, keepdim=True)[0]
        channel_max = torch.max(channel_max, dim=3, keepdim=True)[0]

        avg_out = self.fc_avg(channel_avg.view(b, c)).to(x.device)  
        avg_out = F.relu(avg_out)
        avg_out = self.fc_out_avg(avg_out).view(b, c, 1, 1)

        max_out = self.fc_max(channel_max.view(b, c)).to(x.device) 
        max_out = F.relu(max_out)
        max_out = self.fc_out_max(max_out).view(b, c, 1, 1)

        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class UnetBN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UnetBN, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.conv_block(512, 512)
        self.cbam = CBAM(512)
        self.spatial_attention = SpatialAttention()  

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(512 + 256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),  # Add BatchNorm here
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),  # Add BatchNorm here
        nn.ReLU(inplace=True),
    )


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.cbam(bottleneck)

        dec4 = self.upconv4(bottleneck)
        # spatial_attention = SpatialAttention()  # Instantiate on the forward pass
        dec4 = self.decoder4(torch.cat((dec4, self.spatial_attention(enc4)), dim=1))

        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, self.spatial_attention(enc3)), dim=1))

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, self.spatial_attention(enc2)), dim=1))

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, self.spatial_attention(enc1)), dim=1))

        return self.final_conv(dec1)
