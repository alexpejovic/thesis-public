import argparse
import json
from datetime import datetime, timezone
from logging import DEBUG, Logger
from pathlib import Path
from typing import Any, Type, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import yaml
from feat_encoder.feat_encoder_base import FeatEncoder
from helper import (
    TraceDatasetKey,
    TraceNormalizer,
    get_datagen_config,
    get_vae_config,
    get_voltage_traces,
)
from logger import get_logger, log_commit, log_values
from train import (
    dataloader,
    split_voltage_data,
)
from vae.vae_helper import ACTIVATIONS


# TODO: FINISH THIS
def loss_fn(model, voltage_batch, beta, dt, key, logger: Logger):
    keys = jax.random.split(key, voltage_batch.shape[0])
    feats = jax.vmap(model)(voltage_batch)

    # Reconstruction loss (MSE)
    recon_loss = jnp.mean(jnp.mean((voltage_batch - x_recon) ** 2, axis=1))

    # KL divergence
    kl_loss = jnp.mean(-0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar), axis=1))

    jax.debug.callback(
        log_values, logger, DEBUG, recon_loss=recon_loss, kl_loss=kl_loss
    )

    return recon_loss + beta * kl_loss, (recon_loss, kl_loss)


@eqx.filter_jit
def train_step(
    model, voltage_batch, beta, opt_state, optimizer, dt, model_key, logger: Logger
):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, voltage_batch, beta, dt, model_key, logger
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


jit_loss_vae = eqx.filter_jit(loss_fn)


def val_step(
    model, data_iter, model_key, beta, dt, batches_to_validate, logger: Logger
):
    count = 0
    losses_sum = 0.0
    recon_losses_sum = 0.0
    kl_losses_sum = 0.0
    for i, voltage_batch in enumerate(data_iter):
        if i >= batches_to_validate:
            break
        model_key, model_subkey = jax.random.split(model_key)
        loss, (recon_loss, kl_loss) = jit_loss_vae(
            model, voltage_batch, beta, dt, model_key, logger
        )
        losses_sum += loss.item()
        recon_losses_sum += recon_loss.item()
        kl_losses_sum += kl_loss.item()
        count += 1
        logger.info(f"val_step={i}")

    if count > 0:
        return tuple(s / count for s in (losses_sum, recon_losses_sum, kl_losses_sum))
    else:
        return (0.0,) * 3


def train_vae(
    batch_size: int,
    learning_rate: float,
    epochs: int,
    latent_size: int,
    seed: int,
    beta: float,
    trace_data: TraceDatasetKey,
    vae_type: Type[FeatEncoder],
    normalizer_type: TraceNormalizer,
    dir_path: Path,
    train_split: float,
    model_params: dict[str, Any],
    vals_per_epoch: int,  # The amount of times the model is validated per epoch
    traces_to_validate: int,
    logger: Logger,
    *,
    save_on_val: bool,  # If the model should be saved to disk when running validation
) -> tuple[FeatEncoder, list[float], list[float]]:
    data_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    datagen_config = get_datagen_config()
    voltage_data = get_voltage_traces(trace_data, datagen_config)

    dt = datagen_config["dt"]
    avg_scale = datagen_config["avg_scale"]
    dt = dt * avg_scale

    data_key, data_subkey = jax.random.split(data_key)
    train_indices, val_indices = split_voltage_data(
        voltage_data, train_split, data_subkey
    )

    model = vae_type(voltage_data.shape[1], latent_size, model_key, **model_params)

    steps_per_epoch = len(train_indices) / batch_size
    steps_per_val = int(jnp.ceil(steps_per_epoch / vals_per_epoch))
    batches_to_validate = int(jnp.ceil(traces_to_validate / batch_size))

    optim = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(learning_rate))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_losses = []
    val_losses = []
    val_recon_losses = []
    val_kl_losses = []
    step = 0
    for epoch in range(epochs):
        data_key, data_subkey = jax.random.split(data_key)
        iter_data = dataloader(
            voltage_data,
            batch_size,
            data_subkey,
            normalizer_type,
            indices=train_indices,
            data_type=trace_data,
        )

        logger.info(f"EPOCH={epoch}")

        for voltage_batch in iter_data:
            model_key, model_subkey = jax.random.split(model_key)

            # Saving and validation
            if step % steps_per_val == 0:
                # Saving
                if save_on_val:
                    save_vae(model, dir_path, step)

                # Validation
                logger.info("Doing validation")
                iter_val_data = dataloader(
                    voltage_data,
                    batch_size,
                    data_subkey,
                    normalizer_type,
                    indices=val_indices,
                    shuffle=False,
                    data_type=trace_data,
                )
                val_loss, val_recon_loss, val_kl_loss = val_step(
                    model,
                    iter_val_data,
                    model_key,
                    beta,
                    dt,
                    batches_to_validate,
                    logger,
                )
                val_losses.append(val_loss)
                val_recon_losses.append(val_recon_loss)
                val_kl_losses.append(val_kl_loss)
                logger.info(f"val_loss={val_loss}")

            # Training
            model, opt_state, loss, _ = train_step(
                model, voltage_batch, beta, opt_state, optim, dt, model_subkey, logger
            )
            loss = loss.item()
            train_losses.append(loss)
            logger.info(f"step={step}, loss={loss}")
            step += 1

    # Do a final validation
    if save_on_val:
        save_vae(model, dir_path, step)

    logger.info("Doing validation")
    iter_val_data = dataloader(
        voltage_data,
        batch_size,
        data_subkey,
        normalizer_type,
        indices=val_indices,
        shuffle=False,
        data_type=trace_data,
    )
    val_loss, val_recon_loss, val_kl_loss = val_step(
        model, iter_val_data, model_key, beta, dt, batches_to_validate, logger
    )
    val_losses.append(val_loss)
    val_recon_losses.append(val_recon_loss)
    val_kl_losses.append(val_kl_loss)
    logger.info(f"val_loss={val_loss}")

    return model, train_losses, val_losses, (val_recon_losses, val_kl_losses)


def _parse_args(config: dict) -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    for k, v in config.items():
        if isinstance(v, (dict, list)):
            parser.add_argument(
                f"--{k}", type=json.loads, default=v, help=f"JSON. Default: {v}"
            )
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v, help=f"Default: {v}")

    parser.add_argument(
        "trace_data",
        type=str,
        help=f"Dataset to train on: Possible values: {get_args(TraceDatasetKey)}",
    )

    return parser.parse_args()


def main():
    config = get_vae_config()
    args = _parse_args(config)

    # Create save path and logger
    time_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_base = Path(f"./data/models/vae/{args.trace_data}/{args.vae_type}")

    dir_path = dir_base / time_str
    dir_path.mkdir(parents=True, exist_ok=True)

    config_path = dir_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(vars(args), f, default_flow_style=False)

    logger = get_logger(dir_path / "log", name="train")

    log_commit(logger, "../")

    model_params = args.model_params if args.vae_type == "TimeVAEBase" else {}
    if model_params != {}:
        model_params["activation"] = ACTIVATIONS[args.activation]

    vae, train_losses, val_losses, _ = train_vae(
        args.batch_size,
        args.learning_rate,
        args.epochs,
        args.latent_dim,
        args.seed,
        args.beta,
        args.trace_data,
        MODEL_STRINGS[args.vae_type],
        args.normalizer_type,
        dir_path,
        args.train_split,
        model_params,
        args.vals_per_epoch,
        args.max_valid,
        logger,
        save_on_val=True,
    )

    save_vae_and_link(args.trace_data, vae, time_str, dir_base, dir_path)

    with open(dir_path / "train_losses.npz", "wb") as f:
        jnp.savez(f, train_losses)

    with open(dir_path / "val_losses.npz", "wb") as f:
        jnp.savez(f, val_losses)


if __name__ == "__main__":
    main()
