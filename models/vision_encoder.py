import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for visual observations.
    Maps images (C, H, W) to a latent vector (latent_dim).
    """
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
        hidden_channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2],
        paddings: List[int] = [0, 0, 0, 0],
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size, stride, padding in zip(hidden_channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif activation == "elu":
                layers.append(nn.ELU())
            
            in_channels = out_channels
            
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate output size of conv net to define fc layer
        # Assuming 64x64 input and default params:
        # 64 -> 31 -> 14 -> 6 -> 2 (approx, need to verify)
        # Let's use a dummy input to calculate
        self.feature_dim = self._get_conv_output_size(input_channels)
        
        self.fc = nn.Linear(self.feature_dim, latent_dim)
        
    def _get_conv_output_size(self, input_channels):
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        with torch.no_grad():
            output = self.conv_net(dummy_input)
        return output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input images (B, C, H, W)
        Returns:
            z: Latent vectors (B, latent_dim)
        """
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    """
    Convolutional Decoder for visual observations.
    Maps latent vector (latent_dim) back to images (C, H, W).
    """
    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 3,
        hidden_channels: List[int] = [256, 128, 64, 32],
        kernel_sizes: List[int] = [4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2],
        paddings: List[int] = [0, 0, 0, 0],
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        
        # Reverse lists for decoder
        # Note: hidden_channels should be passed in reverse order of encoder's output
        
        # We need to map latent to the spatial size before the first deconv
        # This is tricky without knowing the exact spatial dims.
        # For now, we'll assume a symmetric architecture to the encoder.
        # If encoder ends at 2x2x256, decoder starts at 2x2x256.
        
        self.initial_spatial_size = 2 # derived from 64 / 2^4 = 4, but let's be careful.
        # Let's assume 64x64 input.
        # L1: 64 -> 32 (k=4, s=2, p=1)
        # L2: 32 -> 16 (k=4, s=2, p=1)
        # L3: 16 -> 8  (k=4, s=2, p=1)
        # L4: 8 -> 4   (k=4, s=2, p=1)
        # So final feature map is 4x4.
        
        # Let's adjust default params to be standard strided convolutions
        # Default in encoder was k=4, s=2, p=0 which reduces size differently.
        # Let's enforce standard padding=1 for k=4, s=2 to halve dimensions exactly.
        
        self.initial_channels = hidden_channels[0]
        self.initial_spatial_size = 4 # Assuming 4 layers of stride 2 on 64x64
        
        self.fc = nn.Linear(latent_dim, self.initial_channels * self.initial_spatial_size * self.initial_spatial_size)
        
        layers = []
        in_channels = self.initial_channels
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(zip(hidden_channels[1:] + [output_channels], kernel_sizes, strides, paddings)):
            is_last = (i == len(hidden_channels) - 1)
            
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
            
            if not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.2))
                elif activation == "elu":
                    layers.append(nn.ELU())
            else:
                layers.append(nn.Sigmoid()) # Output in [0, 1]
            
            in_channels = out_channels
            
        self.deconv_net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            z: Latent vectors (B, latent_dim)
        Returns:
            x: Reconstructed images (B, C, H, W)
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.initial_channels, self.initial_spatial_size, self.initial_spatial_size)
        x = self.deconv_net(x)
        return x

def create_visual_encoder_decoder_pair(
    latent_dim: int = 32,
    input_channels: int = 3,
    image_size: int = 64
) -> Tuple[ConvEncoder, ConvDecoder]:
    """
    Helper to create a matching encoder/decoder pair.
    """
    # Standard architecture for 64x64 images
    hidden_channels = [32, 64, 128, 256]
    kernel_sizes = [4, 4, 4, 4]
    strides = [2, 2, 2, 2]
    paddings = [1, 1, 1, 1] # To ensure exact halving: (H-1)*s - 2p + k = output
    # (64-1)*2 - 2*1 + 4 = 126 + 2 = 128... wait ConvTranspose formula is different.
    # Encoder: floor((H + 2p - k)/s + 1)
    # (64 + 2 - 4)/2 + 1 = 31 + 1 = 32. Correct.
    
    encoder = ConvEncoder(
        input_channels=input_channels,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
    )
    
    decoder = ConvDecoder(
        latent_dim=latent_dim,
        output_channels=input_channels,
        hidden_channels=[256, 128, 64, 32], # Reverse of encoder
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
    )
    
    return encoder, decoder