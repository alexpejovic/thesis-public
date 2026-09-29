import equinox as eqx
from equinox import AbstractVar


class FeatEncoder(eqx.Module):
    base_layer: AbstractVar[eqx.Module]
    final_layer: AbstractVar[eqx.Module]

    def __call__(self, x):
        h = self.base_layer(x)
        final = self.final_layer(h)
        return final
