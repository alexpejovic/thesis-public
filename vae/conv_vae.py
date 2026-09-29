import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from vae.vae_base import VAE, Encoder


class ConvBaseLayer(eqx.Module):
    conv_layers: list[eqx.nn.Conv1d]

    def __init__(
        self,
        input_dim,
        conv_layer_sizes: list[int],
        kernel_size,
        stride,
        activation,
        key,
    ):
        conv_layer_sizes = [1] + conv_layer_sizes
        self.conv_layers = []
        keys = jax.random.split(key, len(conv_layer_sizes) + 1)
        for i in range(len(conv_layer_sizes) - 1):
            in_channels = conv_layer_sizes[i]
            out_channels = conv_layer_sizes[i + 1]
            conv_layer = eqx.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                "SAME",
                key=keys[i],
                dtype=jnp.float32,
            )
            self.conv_layers.append(conv_layer)
            self.conv_layers.append(activation)

    def __call__(self, x: Array):
        x = x.reshape(1, -1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.ravel()
        return x


class TimeVAEBaseEncoder(Encoder):
    base_layer: ConvBaseLayer
    mean_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear
    latent_dim: int

    def __init__(
        self,
        input_dim,
        out_dim,
        conv_layer_sizes,
        kernel_size,
        stride,
        activation,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.base_layer = ConvBaseLayer(
            input_dim,
            conv_layer_sizes,
            kernel_size,
            stride,
            activation,
            key=k1,
        )
        self.latent_dim = int(
            math.ceil(input_dim / 2 ** (len(conv_layer_sizes))) * conv_layer_sizes[-1]
        )
        self.mean_layer = eqx.nn.Linear(
            self.latent_dim,
            out_dim,
            key=k2,
            dtype=jnp.float32,
        )
        self.logvar_layer = eqx.nn.Linear(
            self.latent_dim,
            out_dim,
            key=k3,
            dtype=jnp.float32,
        )


class TimeVAEBaseDecoder(eqx.Module):
    base_layers: eqx.nn.Linear
    conv_layers: list[eqx.nn.ConvTranspose1d]
    final_layer: eqx.nn.Linear

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim,
        conv_layer_sizes,
        kernel_size,
        stride,
        activation,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.base_layers = (
            eqx.nn.Linear(
                input_dim,
                latent_dim,
                key=k1,
                dtype=jnp.float32,
            ),
            activation,
        )
        self.conv_layers = []
        keys = jax.random.split(k2, len(conv_layer_sizes) + 2)
        for i in range(len(conv_layer_sizes) - 1):
            in_channels = conv_layer_sizes[i]
            out_channels = conv_layer_sizes[i + 1]
            conv_layer = eqx.nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding="SAME",
                key=keys[i],
                dtype=jnp.float32,
            )
            self.conv_layers.append(conv_layer)
            self.conv_layers.append(activation)

        self.conv_layers.append(
            eqx.nn.ConvTranspose1d(
                conv_layer_sizes[-1],
                1,
                kernel_size,
                stride,
                padding="SAME",
                key=keys[-2],
                dtype=jnp.float32,
            )
        )
        self.conv_layers.append(activation)

        pre_output_dim = int(
            (latent_dim / conv_layer_sizes[0]) * (2 ** len(conv_layer_sizes))
        )
        self.final_layer = eqx.nn.Linear(
            pre_output_dim,
            output_dim,
            key=keys[-1],
            dtype=jnp.float32,
        )

    def __call__(self, z):
        for layer in self.base_layers:
            z = layer(z)
        z = z.reshape(self.conv_layers[0].in_channels, -1)
        for conv_layer in self.conv_layers:
            z = conv_layer(z)
        z = z.ravel()
        return self.final_layer(z)


class TimeVAEBase(VAE):
    encoder: TimeVAEBaseEncoder
    decoder: TimeVAEBaseDecoder

    def __init__(
        self,
        input_dim,
        latent_dim,
        key,
        *,
        conv_layers,
        kernel_size,
        stride,
        activation,
    ):
        k1, k2 = jax.random.split(key)
        self.encoder = TimeVAEBaseEncoder(
            input_dim,
            latent_dim,
            conv_layers,
            kernel_size,
            stride,
            activation,
            k1,
        )
        self.decoder = TimeVAEBaseDecoder(
            latent_dim,
            input_dim,
            self.encoder.latent_dim,
            conv_layers[::-1],
            kernel_size,
            stride,
            activation,
            k2,
        )
