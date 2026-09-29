import os
from pathlib import Path
from typing import Literal, Type

import equinox as eqx
import jax
import yaml
from helper import ChannelType, TraceDatasetKey
from vae.conv_vae import TimeVAEBase
from vae.linear_vae import LinearVAE
from vae.vae_base import VAE, Encoder

MODELS = [LinearVAE, TimeVAEBase]
MODEL_STRINGS = {vae_type.__name__: vae_type for vae_type in MODELS}

Activation = Literal["relu", "gelu", "elu"]
ACTIVATIONS = {"relu": jax.nn.relu, "gelu": jax.nn.gelu, "elu": jax.nn.elu}


def save_vae(vae: VAE, dir_path: Path, step: int) -> None:
    models_dir = dir_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"model{step:07d}.eqx"
    eqx.tree_serialise_leaves(model_path, vae)


def save_vae_and_link(
    channel_type: ChannelType,
    vae: VAE,
    time_str: str,
    dir_base: Path,
    dir_path: Path,
) -> None:
    model_path = dir_path / "final-model.eqx"
    eqx.tree_serialise_leaves(model_path, vae)

    # Create a simlink to the newest model
    link_path = Path(dir_base / "latest")
    if link_path.exists() and link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(time_str, target_is_directory=True)


def get_available_vae_models(
    data_type: TraceDatasetKey,
    vae_class: Type[VAE],
    model_date: str,
) -> list[str]:
    dir_path = Path(
        f"./data/models/vae/{data_type}/{vae_class.__name__}/{model_date}/models"
    )

    return [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]


def get_vae_config(
    data_type: TraceDatasetKey,
    vae_class: Type[VAE],
    model_date: str,
):
    dir_path = Path(f"./data/models/vae/{data_type}/{vae_class.__name__}/{model_date}")

    config_path = dir_path / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_latest_datestr(data_type: TraceDatasetKey, vae_class: Type[VAE]):
    model_dir = Path(f"./data/models/vae/{data_type}/{vae_class.__name__}/latest")
    model_datestr = model_dir.resolve().name

    return model_datestr


def load_vae(
    data_type: TraceDatasetKey,
    vae_class: Type[VAE],
    input_dim: int,
    model_date: str,
    epoch: str | int | None = None,
) -> VAE:
    dir_path = Path(f"./data/models/vae/{data_type}/{vae_class.__name__}/{model_date}")

    config = get_vae_config(data_type, vae_class, model_date)

    if epoch is None:
        load_path = dir_path / "final-model.eqx"
    elif type(epoch) == int:
        load_path = dir_path / f"models/model{epoch:07d}.eqx"
    else:
        load_path = dir_path / f"models/{epoch}"

    model_params = config["model_params"]
    if model_params != {}:
        model_params["activation"] = ACTIVATIONS[config["activation"]]

    # Note key has no effect since we will load weights right after anyways
    vae = eqx.filter_eval_shape(
        vae_class,
        input_dim,
        config["latent_dim"],
        jax.random.PRNGKey(0),
        **model_params,
    )
    vae = eqx.tree_deserialise_leaves(load_path, vae)
    return vae

def load_named_vae(
    vae_class: Type[VAE],
    input_dim: int,
    model_name: str,
    epoch: str | int | None = None,
) -> VAE:
    dir_path = Path(f"./data/models/vae/named/{model_name}")

    config_path = dir_path / "config.yaml"
    with open(config_path, "r") as f:
        vae_cfg = yaml.safe_load(f)

    if epoch is None:
        load_path = dir_path / "final-model.eqx"
    elif type(epoch) == int:
        load_path = dir_path / f"models/model{epoch:07d}.eqx"
    else:
        load_path = dir_path / f"models/{epoch}"

    model_params = vae_cfg["model_params"]
    if model_params != {}:
        try:
            model_params["activation"] = ACTIVATIONS[vae_cfg["activation"]]
        except KeyError:
            print(f"WARNING: Model {model_name} has no activation in config. Defaulting to relu.")
            model_params["activation"] = jax.nn.relu


    # Note key has no effect since we will load weights right after anyways
    vae = eqx.filter_eval_shape(
        vae_class,
        input_dim,
        vae_cfg["latent_dim"],
        jax.random.PRNGKey(0),
        **model_params,
    )
    vae = eqx.tree_deserialise_leaves(load_path, vae)
    return vae


def load_encoder(
    vae_type: Type[VAE],
    channel_type: ChannelType,
    input_dim: int,
    model_date: str,
    *,
    epoch: int | None = None,
) -> Encoder:
    return load_vae(channel_type, vae_type, input_dim, model_date, epoch).encoder
