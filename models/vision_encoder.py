"""
Vision Encoder for processing image observations.

Convolutional neural networks for encoding visual inputs to latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class VisionEncoder(nn.Module):
    """
    CNN encoder for visual observations.

    Processes images to extract latent representations.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: Tuple[int, int] = (64, 64),
        latent_dim: int = 32,
        channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2],
        batch_norm: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Initialize vision encoder.

        Args:
            image_channels: Number of input channels (1 for grayscale, 3 for RGB)
            image_size: Input image size (H, W)
            latent_dim: Dimension of output latent vector
            channels: Channel sizes for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            strides: Strides for each conv layer
            batch_norm: Whether to use batch normalization
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Build convolutional layers
        conv_layers = []
        in_channels = image_channels

        for i, (out_channels, kernel, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            # Calculate padding to maintain reasonable size reduction
            padding = kernel // 2

            conv_layers.append(
                ConvBlock(
                    in_channels, out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    batch_norm=batch_norm,
                    activation=activation,
                )
            )

            if dropout > 0 and i < len(channels) - 1:  # Don't add dropout after last conv
                conv_layers.append(nn.Dropout2d(dropout))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate size of flattened features
        self.feature_size = self._calculate_feature_size()

        # Fully connected layers to latent
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, latent_dim),
        )

    def _calculate_feature_size(self) -> int:
        """Calculate the size of flattened features after conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, self.image_channels, *self.image_size)
            x = self.conv_layers(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent representation.

        Args:
            x: Image tensor [batch_size, channels, height, width]

        Returns:
            latent: Latent representation [batch_size, latent_dim]
        """
        # Convolutional encoding
        features = self.conv_layers(x)

        # Flatten and map to latent
        latent = self.fc_layers(features)

        return latent


class VisionDecoder(nn.Module):
    """
    CNN decoder for reconstructing images from latent representations.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        image_channels: int = 1,
        image_size: Tuple[int, int] = (64, 64),
        channels: List[int] = [256, 128, 64, 32],
        kernel_sizes: List[int] = [4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2],
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize vision decoder.

        Args:
            latent_dim: Dimension of input latent vector
            image_channels: Number of output channels
            image_size: Target image size (H, W)
            channels: Channel sizes for each deconv layer
            kernel_sizes: Kernel sizes for each deconv layer
            strides: Strides for each deconv layer
            batch_norm: Whether to use batch normalization
            activation: Activation function
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size

        # Initial size after reshaping latent
        self.initial_size = (channels[0], image_size[0] // (2 ** len(strides)),
                           image_size[1] // (2 ** len(strides)))

        # Fully connected to initial feature map
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.initial_size[0] * self.initial_size[1] * self.initial_size[2]),
            nn.ReLU(),
        )

        # Build deconvolutional layers
        deconv_layers = []

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            kernel = kernel_sizes[i]
            stride = strides[i]
            padding = kernel // 2
            output_padding = stride - 1

            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )

            if batch_norm:
                deconv_layers.append(nn.BatchNorm2d(out_channels))

            if activation == "relu":
                deconv_layers.append(nn.ReLU(inplace=True))
            elif activation == "elu":
                deconv_layers.append(nn.ELU(inplace=True))

        # Final layer to image
        deconv_layers.append(
            nn.ConvTranspose2d(
                channels[-1], image_channels,
                kernel_size=kernel_sizes[-1],
                stride=strides[-1],
                padding=kernel_sizes[-1] // 2,
                output_padding=strides[-1] - 1,
            )
        )
        deconv_layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.deconv_layers = nn.Sequential(*deconv_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.

        Args:
            latent: Latent representation [batch_size, latent_dim]

        Returns:
            image: Reconstructed image [batch_size, channels, height, width]
        """
        # Map latent to initial feature map
        features = self.fc_layers(latent)

        # Reshape to spatial dimensions
        features = features.view(-1, *self.initial_size)

        # Deconvolutional decoding
        image = self.deconv_layers(features)

        return image


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels) if batch_norm else None
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels) if batch_norm else None

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        if self.bn1:
            out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)

        out = out + residual
        out = self.activation(out)

        return out


class ResNetVisionEncoder(nn.Module):
    """
    ResNet-style vision encoder for more complex visual tasks.
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: Tuple[int, int] = (64, 64),
        latent_dim: int = 32,
        channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 2,
        batch_norm: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Initialize ResNet vision encoder.

        Args:
            image_channels: Number of input channels
            image_size: Input image size
            latent_dim: Dimension of output latent vector
            channels: Channel sizes for each stage
            num_residual_blocks: Number of residual blocks per stage
            batch_norm: Whether to use batch normalization
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Initial convolution
        layers = [
            nn.Conv2d(image_channels, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(channels[0]) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        # Residual stages
        in_channels = channels[0]
        for out_channels in channels[1:]:
            # Downsampling convolution
            layers.append(
                ConvBlock(
                    in_channels, out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    batch_norm=batch_norm,
                    activation=activation,
                )
            )

            # Residual blocks
            for _ in range(num_residual_blocks):
                layers.append(
                    ResidualBlock(
                        out_channels,
                        batch_norm=batch_norm,
                        activation=activation,
                    )
                )

            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected
        self.fc = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent representation.

        Args:
            x: Image tensor [batch_size, channels, height, width]

        Returns:
            latent: Latent representation [batch_size, latent_dim]
        """
        # Convolutional encoding
        features = self.conv_layers(x)

        # Global pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)

        # Map to latent
        latent = self.fc(features)

        return latent


def create_vision_encoder(
    architecture: str = "simple",
    image_channels: int = 1,
    image_size: Tuple[int, int] = (64, 64),
    latent_dim: int = 32,
) -> nn.Module:
    """
    Factory function to create vision encoders.

    Args:
        architecture: Type of encoder ("simple", "resnet")
        image_channels: Number of input channels
        image_size: Input image size
        latent_dim: Dimension of latent representation

    Returns:
        Vision encoder module
    """
    if architecture == "simple":
        return VisionEncoder(
            image_channels=image_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            channels=[32, 64, 128],
            kernel_sizes=[4, 4, 4],
            strides=[2, 2, 2],
        )
    elif architecture == "resnet":
        return ResNetVisionEncoder(
            image_channels=image_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            channels=[32, 64, 128],
            num_residual_blocks=2,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")