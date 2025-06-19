import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.main_branch = nn.Conv2d(
            in_channels, 
            out_channels - 3,  
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        
        self.ext_branch = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        
        
        out = torch.cat((main, ext), dim=1)
        
        
        out = self.batch_norm(out)
        out = self.prelu(out)
        
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 dropout_prob=0.1, downsampling=False, 
                 asymmetric=False, dilated=False, 
                 dilation_rate=1, regularlizer_prob=0.1,
                 upsampling=False):
        super().__init__()
        
        internal_channels = in_channels // 4  
        self.upsampling = upsampling
        
        
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU()
        )
        
        
        if downsampling:
            self.conv2x2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels, 
                    internal_channels, 
                    kernel_size=2, 
                    stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )
        elif upsampling:
            self.conv2x2 = nn.Sequential(
                nn.ConvTranspose2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )
        elif asymmetric:
            self.conv2x2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(5, 1),
                    padding=(2, 0),
                    bias=False
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, 5),
                    padding=(0, 2),
                    bias=False
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )
        elif dilated:
            self.conv2x2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                    bias=False
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )
        else:  
            self.conv2x2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU()
            )
        
        
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob)
        )
        
        
        self.regularizer = nn.Dropout2d(p=regularlizer_prob)
        
        
        if downsampling:
            self.resample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif upsampling:
            self.resample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.resample = None
        
        self.out_prelu = nn.PReLU()
    
    def forward(self, x):
        residual = x
        
        
        main = self.conv1x1_1(x)
        main = self.conv2x2(main)
        main = self.conv1x1_2(main)
        main = self.regularizer(main)
        
        
        if self.resample is not None:
            residual = self.resample(residual)
        
        
        out = main + residual
        
        return self.out_prelu(out)


class ENet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        
        self.initial = InitialBlock(3, 16)
        
        
        self.bottleneck1_0 = Bottleneck(16, 64, downsampling=True, dropout_prob=0.01)
        self.bottleneck1_1 = Bottleneck(64, 64)
        self.bottleneck1_2 = Bottleneck(64, 64)
        self.bottleneck1_3 = Bottleneck(64, 64)
        self.bottleneck1_4 = Bottleneck(64, 64)
        
        
        self.bottleneck2_0 = Bottleneck(64, 128, downsampling=True)
        self.bottleneck2_1 = Bottleneck(128, 128)
        self.bottleneck2_2 = Bottleneck(128, 128, dilated=True, dilation_rate=2)
        self.bottleneck2_3 = Bottleneck(128, 128, asymmetric=True)
        self.bottleneck2_4 = Bottleneck(128, 128, dilated=True, dilation_rate=4)
        self.bottleneck2_5 = Bottleneck(128, 128)
        self.bottleneck2_6 = Bottleneck(128, 128, dilated=True, dilation_rate=8)
        self.bottleneck2_7 = Bottleneck(128, 128, asymmetric=True)
        self.bottleneck2_8 = Bottleneck(128, 128, dilated=True, dilation_rate=16)
        
        
        self.bottleneck3_1 = Bottleneck(128, 128)
        self.bottleneck3_2 = Bottleneck(128, 128, dilated=True, dilation_rate=2)
        self.bottleneck3_3 = Bottleneck(128, 128, asymmetric=True)
        self.bottleneck3_4 = Bottleneck(128, 128, dilated=True, dilation_rate=4)
        self.bottleneck3_5 = Bottleneck(128, 128)
        self.bottleneck3_6 = Bottleneck(128, 128, dilated=True, dilation_rate=8)
        self.bottleneck3_7 = Bottleneck(128, 128, asymmetric=True)
        self.bottleneck3_8 = Bottleneck(128, 128, dilated=True, dilation_rate=16)
        
        
        self.bottleneck4_0 = Bottleneck(128, 64, upsampling=True)
        self.bottleneck4_1 = Bottleneck(64, 64)
        self.bottleneck4_2 = Bottleneck(64, 64)
        
        
        self.bottleneck5_0 = Bottleneck(64, 16, upsampling=True)
        self.bottleneck5_1 = Bottleneck(16, 16)
        
        
        self.fullconv = nn.ConvTranspose2d(
            16, num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
    
    def forward(self, x):
        
        x = self.initial(x)
        
        
        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)
        
        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)
        
        
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        x = self.bottleneck3_8(x)
        
        x = self.bottleneck4_0(x)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)
        
        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)
        
        
        x = self.fullconv(x)
        
        return x