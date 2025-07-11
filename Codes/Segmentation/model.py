import torch
import torch.nn as nn
from monai.networks.nets import UNet, VNet
from monai.networks.layers import Norm

def get_model(
    architecture: str = "vnet",
    in_channels: int = 4,
    out_channels: int = 5,
    spatial_dims: int = 3,
) -> nn.Module:
    """
    Returns the segmentation model suitable for BraTS 2025 task.

    Args:
        architecture (str): "unet", "vnet", or "custom".
        in_channels (int): Number of input MRI channels (BraTS: 4).
        out_channels (int): Number of segmentation classes (BraTS: 5).
        spatial_dims (int): 2 or 3 (BraTS: 3).
    Returns:
        nn.Module: Instantiated segmentation model.
    """

    if architecture.lower() == "unet":
        # MONAI 3D UNet, tuned for BraTS
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm='INSTANCE'
        )
    elif architecture.lower() == "vnet":
        # MONAI VNet, robust for multi-class brain tumor segmentation
        return VNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob_up=(0.2,0.2),
            dropout_prob_down=0.2,
            act=("PReLU", {"init": 0.2})
        )
    elif architecture.lower() == "custom":
        # Custom: Residual 3D CNN with deep feature extraction and skip connections
        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
                self.bn1 = nn.InstanceNorm3d(out_ch)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
                self.bn2 = nn.InstanceNorm3d(out_ch)
                self.downsample = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
            def forward(self, x):
                identity = self.downsample(x)
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += identity
                return self.relu(out)
        class CustomSegNet(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.enc1 = ResidualBlock(in_channels, 32)
                self.enc2 = ResidualBlock(32, 64)
                self.pool1 = nn.MaxPool3d(2)
                self.enc3 = ResidualBlock(64, 128)
                self.pool2 = nn.MaxPool3d(2)
                self.enc4 = ResidualBlock(128, 256)
                self.pool3 = nn.MaxPool3d(2)
                self.center = ResidualBlock(256, 320)
                self.up3 = nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2)
                self.dec3 = ResidualBlock(512, 128)
                self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
                self.dec2 = ResidualBlock(128, 64)
                self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
                self.dec1 = ResidualBlock(64, 32)
                self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(e1)
                p1 = self.pool1(e2)
                e3 = self.enc3(p1)
                p2 = self.pool2(e3)
                e4 = self.enc4(p2)
                p3 = self.pool3(e4)
                center = self.center(p3)
                u3 = self.up3(center)
                d3 = self.dec3(torch.cat([u3, e4], dim=1))
                u2 = self.up2(d3)
                d2 = self.dec2(torch.cat([u2, e3], dim=1))
                u1 = self.up1(d2)
                d1 = self.dec1(torch.cat([u1, e2], dim=1))
                out = self.out_conv(d1)
                return out
        return CustomSegNet(in_channels, out_channels)
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose from 'unet', 'vnet', or 'custom'."
        )

# Example usage
if __name__ == "__main__":
    # Architecture can be parsed from config.yaml or CLI args
    model = get_model("vnet", in_channels=4, out_channels=5, spatial_dims=3)
    print(model)
