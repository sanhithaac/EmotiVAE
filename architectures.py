"""
architectures.py
-----------------
Neural network architecture definitions for EmotiVAE.
Implements a face image encoder, decoder, and the full
Conditional Variational Autoencoder (ConditionalVAE) pipeline.
"""

import torch
import torch.nn as nn


class GaussianDistribution(torch.distributions.Distribution):
    """Parameterised Gaussian used for the latent space."""

    def __init__(self, mean, log_std):
        self.mean = mean
        self.std = log_std.exp()

    def sample_noise(self):
        return torch.rand_like(self.std)

    def rsample(self):
        noise = self.sample_noise()
        return self.mean + self.std * noise

    def log_prob(self, z_sample):
        normal = torch.distributions.Normal(
            loc=self.mean, scale=self.std, validate_args=False
        )
        return normal.log_prob(z_sample)


def _init_weights(module):
    """Apply Kaiming initialisation to Conv and Linear layers."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FaceEncoder(nn.Module):
    """Convolutional encoder that maps an image to a latent vector."""

    def __init__(self, img_shape, latent_dim, variational=False):
        super(FaceEncoder, self).__init__()
        self.img_shape = img_shape
        self.channels = img_shape[1]
        self.height = img_shape[2]
        self.width = img_shape[3]
        self.latent_dim = latent_dim
        self.variational = variational

        self.conv_block = nn.Sequential(
            # e.g. [3, 50, 50]
            nn.Conv2d(self.channels, 32, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # e.g. [32, 25, 25]
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            # e.g. [64, 13, 13]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            # e.g. [128, 7, 7]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            # e.g. [256, 4, 4]
        )

        self.flatten_layer = nn.Flatten(start_dim=1)

        output_dim = 2 * self.latent_dim if self.variational else self.latent_dim
        self.fc_block = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ELU(),
            nn.Dropout(p=0.15),
            nn.Linear(1024, 256),
            nn.ELU(),
            nn.Linear(256, output_dim),
        )

        self.apply(_init_weights)

    def forward(self, img_tensor):
        features = self.conv_block(img_tensor)
        flat = self.flatten_layer(features)
        latent_vec = self.fc_block(flat)
        return latent_vec


class FaceDecoder(nn.Module):
    """Transpose-convolutional decoder that reconstructs an image from a latent vector."""

    def __init__(self, target_shape, latent_dim):
        super(FaceDecoder, self).__init__()
        self.target_shape = target_shape
        self.out_channels = target_shape[1]
        self.out_h = target_shape[2]
        self.out_w = target_shape[3]
        self.latent_dim = latent_dim

        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1024),
            nn.ELU(),
            nn.Linear(1024, 4096),
            nn.ELU(),
        )

        self.reshape_layer = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))

        self.deconv_block = nn.Sequential(
            # [256, 4, 4] -> [128, 7, 7]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            # [128, 7, 7] -> [64, 13, 13]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # [64, 13, 13] -> [32, 25, 25]
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # [32, 25, 25] -> [3, 50, 50]
            nn.ConvTranspose2d(32, self.out_channels, kernel_size=6, stride=2, padding=2),
        )

        self.apply(_init_weights)

    def forward(self, z_vec):
        h = self.fc_block(z_vec)
        h = self.reshape_layer(h)
        h = self.deconv_block(h)
        out = torch.sigmoid(h)
        return out


class BasicAutoencoder(nn.Module):
    """Deterministic autoencoder (AE) baseline."""

    def __init__(self, img_shape, latent_dim):
        super(BasicAutoencoder, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.encoder = FaceEncoder(img_shape=img_shape, latent_dim=latent_dim)
        self.decoder = FaceDecoder(target_shape=img_shape, latent_dim=latent_dim)

    def forward(self, img_tensor):
        z_vec = self.encoder(img_tensor)
        reconstruction = self.decoder(z_vec)
        return reconstruction


class VariationalAE(nn.Module):
    """Standard Variational Autoencoder (VAE)."""

    def __init__(self, img_shape, latent_dim):
        super(VariationalAE, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * self.latent_dim]))
        )

        self.encoder = FaceEncoder(
            img_shape=img_shape, latent_dim=latent_dim, variational=True
        )
        self.decoder = FaceDecoder(
            target_shape=img_shape, latent_dim=latent_dim
        )

    def get_posterior(self, img_tensor):
        h = self.encoder(img_tensor)
        mean, log_std = torch.chunk(h, chunks=2, dim=-1)
        return GaussianDistribution(mean=mean, log_std=log_std)

    def get_prior(self, n_samples):
        params = self.prior_params.expand(n_samples, *self.prior_params.shape[-1:])
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        return GaussianDistribution(mean=mean, log_std=log_std)

    def reconstruct(self, z_sample):
        return self.decoder(z_sample)

    def draw_from_prior(self, n_samples):
        prior = self.get_prior(n_samples=n_samples)
        z_sample = prior.rsample()
        return self.reconstruct(z_sample), z_sample

    def forward(self, img_tensor):
        n_samples = img_tensor.shape[0]
        q_z = self.get_posterior(img_tensor)
        p_z = self.get_prior(n_samples=n_samples)
        z_sample = q_z.rsample()
        reconstruction = self.reconstruct(z_sample)
        return {
            "p_z": p_z,
            "q_z": q_z,
            "z_sample": z_sample,
            "original": img_tensor,
            "reconstructed": reconstruction,
        }


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE).
    Conditions the decoder on an additional scalar label
    (expression intensity score) concatenated to the latent vector.
    """

    def __init__(self, img_shape, latent_dim):
        super(ConditionalVAE, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.register_buffer(
            "prior_params", torch.zeros(torch.Size([1, 2 * self.latent_dim]))
        )

        self.encoder = FaceEncoder(
            img_shape=img_shape, latent_dim=latent_dim, variational=True
        )
        # Decoder receives latent_dim + 1 (the condition scalar)
        self.decoder = FaceDecoder(
            target_shape=img_shape, latent_dim=latent_dim + 1
        )

    def get_posterior(self, img_tensor):
        h = self.encoder(img_tensor)
        mean, log_std = torch.chunk(h, chunks=2, dim=-1)
        return GaussianDistribution(mean=mean, log_std=log_std)

    def get_prior(self, n_samples):
        params = self.prior_params.expand(n_samples, *self.prior_params.shape[-1:])
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        return GaussianDistribution(mean=mean, log_std=log_std)

    def reconstruct(self, z_conditioned):
        return self.decoder(z_conditioned)

    def draw_from_prior(self, n_samples, condition):
        prior = self.get_prior(n_samples=n_samples)
        z_sample = prior.rsample()
        if z_sample.is_cuda:
            cond_vec = condition * torch.unsqueeze(torch.ones(n_samples), 1).cuda()
        else:
            cond_vec = condition * torch.unsqueeze(torch.ones(n_samples), 1)
        z_conditioned = torch.cat((z_sample, cond_vec), dim=1)
        return self.reconstruct(z_conditioned), z_conditioned

    def forward(self, img_tensor, expression_score):
        n_samples = img_tensor.shape[0]
        q_z = self.get_posterior(img_tensor)
        p_z = self.get_prior(n_samples=n_samples)
        z_sample = q_z.rsample()
        cond_vec = torch.unsqueeze(expression_score, dim=1)
        z_conditioned = torch.cat((z_sample, cond_vec), dim=1).float()
        reconstruction = self.reconstruct(z_conditioned)
        return {
            "p_z": p_z,
            "q_z": q_z,
            "z_sample": z_sample,
            "z_conditioned": z_conditioned,
            "original": img_tensor,
            "reconstructed": reconstruction,
        }
