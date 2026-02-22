"""Generative model implementations"""

from typing import Callable, Tuple

import torch
import torch.nn as nn


# Generative model configurations
GENERATIVE_MODELS = {
    "vae": {
        "name": "VAE",
        "params": "~10M",
        "input_size": (64, 64),
        "latent_dim": 128,
        "description": "Variational Autoencoder",
    },
    "unet": {
        "name": "U-Net",
        "params": "~30M",
        "input_size": (128, 128),
        "description": "U-Net for image segmentation",
    },
    "autoencoder": {
        "name": "Autoencoder",
        "params": "~5M",
        "input_size": (64, 64),
        "latent_dim": 256,
        "description": "Simple convolutional autoencoder",
    },
}


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, latent_dim: int = 128, input_channels: int = 3):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return {"logits": self.decode(z), "mu": mu, "log_var": log_var}


class UNet(nn.Module):
    """U-Net architecture for image segmentation"""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))

        return {"logits": self.final(d3)}


class Autoencoder(nn.Module):
    """Simple convolutional autoencoder"""

    def __init__(self, latent_dim: int = 256, input_channels: int = 3):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return {"logits": self.decoder(z)}


def get_generative_model(model_name: str) -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """
    Get a generative model by name.

    Args:
        model_name: Name of the model

    Returns:
        Tuple of (model, input_shape, get_input_fn)
    """
    if model_name == "vae":
        return get_vae_model()
    elif model_name == "unet":
        return get_unet_model()
    elif model_name == "autoencoder":
        return get_autoencoder_model()
    else:
        raise ValueError(f"Unknown generative model: {model_name}")


def get_vae_model() -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """Get VAE model"""
    config = GENERATIVE_MODELS["vae"]
    model = VAE(latent_dim=config["latent_dim"])
    input_size = config["input_size"]

    input_shape = (1, 3, input_size[0], input_size[1])

    def get_input_fn(batch_size: int, device: str):
        return torch.randn(
            batch_size, 3, input_size[0], input_size[1],
            device=device, dtype=torch.float32
        )

    return model, input_shape, get_input_fn


def get_unet_model() -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """Get U-Net model"""
    config = GENERATIVE_MODELS["unet"]
    model = UNet(in_channels=3, out_channels=1)
    input_size = config["input_size"]

    input_shape = (1, 3, input_size[0], input_size[1])

    def get_input_fn(batch_size: int, device: str):
        return torch.randn(
            batch_size, 3, input_size[0], input_size[1],
            device=device, dtype=torch.float32
        )

    return model, input_shape, get_input_fn


def get_autoencoder_model() -> Tuple[nn.Module, Tuple[int, ...], Callable]:
    """Get Autoencoder model"""
    config = GENERATIVE_MODELS["autoencoder"]
    model = Autoencoder(latent_dim=config["latent_dim"])
    input_size = config["input_size"]

    input_shape = (1, 3, input_size[0], input_size[1])

    def get_input_fn(batch_size: int, device: str):
        return torch.randn(
            batch_size, 3, input_size[0], input_size[1],
            device=device, dtype=torch.float32
        )

    return model, input_shape, get_input_fn
