import math

import equinox as eqx
import jax
import jax.numpy as jnp
from feat_encoder_base import FeatEncoder
from vae.conv_vae import ConvBaseLayer


class TimeVAEFeatEncoder(FeatEncoder):
    base_layer: ConvBaseLayer
    final_layer: eqx.nn.Linear
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
        self.final_layer = eqx.nn.Linear(
            self.latent_dim,
            out_dim,
            key=k2,
            dtype=jnp.float32,
        )
