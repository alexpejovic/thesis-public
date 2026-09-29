import equinox as eqx
import jax
from vae.vae_base import VAE, Encoder


class LinearEncoder(Encoder):
    base_layer: eqx.nn.MLP
    mean_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear

    def __init__(self, input_dim, out_dim, key, *, mlp_hidden_dim=128, latent_dim=128):
        k1, k2, k3 = jax.random.split(key, 3)
        self.base_layer = eqx.nn.MLP(
            input_dim, latent_dim, mlp_hidden_dim, depth=2, key=k1
        )
        self.mean_layer = eqx.nn.Linear(latent_dim, out_dim, key=k2)
        self.logvar_layer = eqx.nn.Linear(latent_dim, out_dim, key=k3)


class LinearDecoder(eqx.Module):
    base_layer: eqx.nn.MLP
    output_layer: eqx.nn.Linear

    def __init__(
        self, input_dim, output_dim, key, *, mlp_hidden_dim=128, latent_dim=128
    ):
        k1, k2 = jax.random.split(key)
        self.base_layer = eqx.nn.MLP(
            input_dim, latent_dim, mlp_hidden_dim, depth=2, key=k1
        )
        self.output_layer = eqx.nn.Linear(latent_dim, output_dim, key=k2)

    def __call__(self, z):
        h = self.base_layer(z)
        return self.output_layer(h)


class LinearVAE(VAE):
    encoder: LinearEncoder
    decoder: LinearDecoder

    def __init__(self, input_dim, latent_dim, key):
        k1, k2 = jax.random.split(key)
        self.encoder = LinearEncoder(input_dim, latent_dim, k1)
        self.decoder = LinearDecoder(latent_dim, input_dim, k2)
