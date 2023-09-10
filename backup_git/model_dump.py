class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Downsampling
        self.conv1 = conv_block(2, 128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_block(128, 256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv_block(256, 512)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Global Average Pooling and a Conv layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_conv = nn.Conv2d(512, 1024, kernel_size=1)  # Adjusting the channel dimension
        self.gap_upsample = nn.Upsample(scale_factor=(16, 16))  # Upsampling the GAP output

        # Bridge
        self.conv4 = conv_block(1024, 1024)

        # Upsampling
        self.upconv6 = upconv_block(1024, 512)
        self.conv6 = conv_block(1024, 512)
        self.upconv7 = upconv_block(512, 256)
        self.conv7 = conv_block(512, 256)
        self.upconv8 = upconv_block(256, 128)
        self.conv8 = conv_block(256, 128)

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1)
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        # Applying GAP and Upsampling
        gap = self.global_avg_pool(x6)
        x7 = self.gap_conv(gap)
        x7_upsampled = self.gap_upsample(x7)

        x8 = self.conv4(x7_upsampled)

        x9 = self.upconv6(x8)
        x10 = torch.cat([x9, x5], dim=1)
        x11 = self.conv6(x10)

        x12 = self.upconv7(x11)
        x13 = torch.cat([x12, x3], dim=1)
        x14 = self.conv7(x13)

        x15 = self.upconv8(x14)
        x16 = torch.cat([x15, x1], dim=1)
        x17 = self.conv8(x16)

        out = self.output(x17)

        return out
    


class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Downsampling
        self.conv1 = conv_block(2, 128)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = conv_block(128, 256)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = conv_block(256, 512)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv4 = conv_block(512, 1024)
        self.pool4 = nn.AvgPool2d(2, 2)

        # Bridge
        self.conv5 = conv_block(1024, 2048)
        self.conv6 = conv_block(2048, 2048)

        # Upsampling
        self.upconv7 = upconv_block(2048, 1024)
        self.conv7 = conv_block(2048, 1024)
        self.upconv8 = upconv_block(1024, 512)
        self.conv8 = conv_block(1024, 512)
        self.upconv9 = upconv_block(512, 256)
        self.conv9 = conv_block(512, 256)
        self.upconv10 = upconv_block(256, 128)
        self.conv10 = conv_block(256, 128)

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1)
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        x10 = self.conv6(x9)

        x11 = self.upconv7(x10)
        x12 = torch.cat([x11, x7], dim=1)
        x13 = self.conv7(x12)

        x14 = self.upconv8(x13)
        x15 = torch.cat([x14, x5], dim=1)
        x16 = self.conv8(x15)

        x17 = self.upconv9(x16)
        x18 = torch.cat([x17, x3], dim=1)
        x19 = self.conv9(x18)

        x20 = self.upconv10(x19)
        x21 = torch.cat([x20, x1], dim=1)
        x22 = self.conv10(x21)

        out = self.output(x22)

        return out
    

class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()

        def conv_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Downsampling
        self.conv1 = conv_block(2, 128, stride=2) # Strided Convolution Replaces Pooling
        self.conv2 = conv_block(128, 256, stride=2) # Strided Convolution Replaces Pooling
        self.conv3 = conv_block(256, 512, stride=2) # Strided Convolution Replaces Pooling
        self.conv4 = conv_block(512, 1024, stride=2) # Strided Convolution Replaces Pooling

        # Bridge
        self.conv5 = conv_block(1024, 2048)
        self.conv6 = conv_block(2048, 2048)

        # Upsampling
        self.upconv7 = upconv_block(2048, 1024)
        self.conv7 = conv_block(1536, 1024)
        self.upconv8 = upconv_block(1024, 512)
        self.conv8 = conv_block(768, 512)
        self.upconv9 = upconv_block(512, 256)
        self.conv9 = conv_block(384, 256)
        self.upconv10 = upconv_block(256, 128)
        self.conv10 = conv_block(128, 64)

        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1)

        x1 = self.conv1(x)  # Output of Strided Convolution, spatial size: 64x64
        x2 = self.conv2(x1) # Output of Strided Convolution, spatial size: 32x32
        x3 = self.conv3(x2) # Output of Strided Convolution, spatial size: 16x16
        x4 = self.conv4(x3) # Output of Strided Convolution, spatial size: 8x8

        x5 = self.conv5(x4) # Spatial size remains: 8x8
        x6 = self.conv6(x5) # Spatial size remains: 8x8

        x7 = self.upconv7(x6)  # Upsampled, spatial size: 16x16
        x8 = torch.cat([x7, x3], dim=1) # Match spatial size with x3
        x9 = self.conv7(x8)

        x10 = self.upconv8(x9)  # Upsampled, spatial size: 32x32
        x11 = torch.cat([x10, x2], dim=1) # Match spatial size with x2
        x12 = self.conv8(x11)

        x13 = self.upconv9(x12)  # Upsampled, spatial size: 64x64
        x14 = torch.cat([x13, x1], dim=1) # Match spatial size with x1
        x15 = self.conv9(x14)

        x16 = self.upconv10(x15) # Upsampled, spatial size: 128x128
        # Here, we don't have a corresponding feature map from the encoder to concatenate
        x17 = self.conv10(x16) 

        out = self.output(x17)

        return out
    
 #%%
 class seg_UNet(nn.Module):
    def __init__(self):
        super(seg_UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate=0.5):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )

        def upconv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Downsampling
        self.conv1 = conv_block(4, 128)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = conv_block(128, 256)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = conv_block(256, 512)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv4 = conv_block(512, 1024)
        self.pool4 = nn.AvgPool2d(2, 2)

        # Bridge
        self.conv5 = conv_block(1024, 2048)
        self.conv6 = conv_block(2048, 2048)

        # Upsampling
        self.upconv7 = upconv_block(2048, 1024)
        self.conv7 = conv_block(2048, 1024)
        self.dropout7 = nn.Dropout(0.3)
        self.upconv8 = upconv_block(1024, 512)
        self.conv8 = conv_block(1024, 512)
        self.dropout8 = nn.Dropout(0.3)
        self.upconv9 = upconv_block(512, 256)
        self.conv9 = conv_block(512, 256)
        self.dropout9 = nn.Dropout(0.3)
        self.upconv10 = upconv_block(256, 128)
        self.conv10 = conv_block(256, 128)

        self.output = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2, img3, img4):
        x = torch.cat((img1, img2, img3, img4), dim=1)
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        x10 = self.conv6(x9)

        x11 = self.upconv7(x10)
        x12 = torch.cat([x11, x7], dim=1)
        x13 = self.conv7(x12)

        x14 = self.upconv8(x13)
        x15 = torch.cat([x14, x5], dim=1)
        x16 = self.conv8(x15)

        x17 = self.upconv9(x16)
        x18 = torch.cat([x17, x3], dim=1)
        x19 = self.conv9(x18)

        x20 = self.upconv10(x19)
        x21 = torch.cat([x20, x1], dim=1)
        x22 = self.conv10(x21)

        out = self.output(x22)

        return out