import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar


class Encoder(eqx.Module):
    base_layer: AbstractVar[eqx.Module]
    mean_layer: AbstractVar[eqx.Module]
    logvar_layer: AbstractVar[eqx.Module]

    def __call__(self, x):
        h = self.base_layer(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar


class VAE(eqx.Module):
    encoder: AbstractVar[Encoder]
    decoder: AbstractVar[eqx.Module]

    def reparameterize(self, mean, logvar, key):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        return mean + eps * std

    def __call__(self, x, key):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar, key)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

    def predict(self, x):
        mean, _ = self.encoder(x)
        return self.decoder(mean)
