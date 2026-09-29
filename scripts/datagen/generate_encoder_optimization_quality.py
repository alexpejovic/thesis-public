import argparse
import os
import sys
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Literal, Type

import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import yaml
from jaxley import Compartment

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from helper import (
    ChannelType,
    FeatureLoss,
    compress_traces,
    get_datagen_config,
    get_param_bounds,
    get_trace_normal_and_denormal,
    insert_channels,
    set_const_params,
)
from logger import get_logger, log_commit
from train_comp import (
    _backward_transform_params,
    _forward_transform_params,
    _get_transform_list,
    get_traces_to_match,
    set_frozen_params,
)
from vae.vae_base import VAE
from vae.vae_helper import (
    MODEL_STRINGS,
    get_latest_datestr,
    get_vae_config,
    load_encoder,
)

jax.config.update("jax_enable_x64", True)


def _jaxley_polyak(lr_scale, beta):
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, *, value, **extra):
        del params, extra
        gnorm = optax.global_norm(updates)
        # `where` rather than `gnorm + eps`: exact wherever the gradient is
        # non-zero, and finite AT the optimum, where the reference loop's 0/0
        # would hand a NaN to the trailing zero_nans instead of standing still.
        denom = jnp.where(gnorm > 0, gnorm**beta, 1.0)
        step = lr_scale * value
        return jax.tree.map(lambda g: -step * g / denom, updates), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


Optimizer = Literal["rmsprop", "adam", "polyak", "jaxley_polyak"]

OPTIMIZERS = {
    "rmsprop": optax.rmsprop,
    "adam": optax.adam,
    "polyak": optax.polyak_sgd,
    "jaxley_polyak": _jaxley_polyak,
}

INIT_SEEDS = range(100)


def _get_encoder_optimization_distance(
    channel_type: ChannelType,
    seed: int,
    vae_type: Type[VAE],
    encoder_datestr: str,
    encoder_epoch: int | None,
    freeze_params: list[str],
    feat_loss_type: FeatureLoss | Literal["mse"],
    learning_rate: float,
    optimizer: Callable[[Any], optax.GradientTransformationExtraArgs],
    start_dist: float,
    dist_stepsize: float,
    base_train_steps: int,
    train_steps_scaler: int,
    failure_distance: float,
    logger: Logger,
    new_distance_score: bool = True,
    rand_init_bounds: tuple[float, float] = (-3.0, 3.0),
):
    """For a given encoder and respective loss, return the first distance away
    from the ideals where the optimizer cannot converge.
    """
    comp = Compartment()
    comp = insert_channels(comp, channel_type)

    config = get_datagen_config()
    dt = config["dt"]
    t_max = config["t_max"]
    current_params = config["current"]
    t_filter = config["t_filter"]
    avg_scale = config["avg_scale"]

    comp = set_const_params(config["param_consts"], channel_type, comp)
    comp.record("v")

    ideal_data = get_traces_to_match(channel_type, config, 0)

    amps = ideal_data[1]

    param_bounds = get_param_bounds(config["param_bounds"], channel_type)
    training_bounds = {k: v for k, v in param_bounds.items() if k not in freeze_params}
    freeze_bounds = {k: v for k, v in param_bounds.items() if k in freeze_params}
    transform_list = _get_transform_list(training_bounds)

    ideal_params_dict = {}
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    for param_name, bounds in param_bounds.items():
        key, subkey = jax.random.split(key)
        bound_center = (bounds[0] + bounds[1]) / 2
        q1 = (bounds[0] + bound_center) / 2
        q2 = (bound_center + bounds[1]) / 2
        val = jax.random.uniform(subkey, minval=q1, maxval=q2)
        ideal_params_dict[param_name] = val

    comp = set_frozen_params(freeze_bounds, comp, ideal_params_dict)
    comp.init_states(delta_t=dt)

    transform_list = _get_transform_list(training_bounds)

    transformed_ideals = {
        p[0]: float(p[1])
        for p in zip(
            training_bounds.keys(),
            _backward_transform_params(
                [v for k, v in ideal_params_dict.items() if k not in freeze_params],
                transform_list,
            ),
        )
    }

    def simulate_network(params, amp):
        """Simulates the network, returning the voltage traces of the output neurons"""
        params = _forward_transform_params(params, transform_list)

        current = jx.step_current(
            i_delay=current_params["delay"],
            i_dur=current_params["duration"],
            i_amp=amp,
            delta_t=dt,
            t_max=t_max,
        )

        param_state = None
        for i, param in enumerate(training_bounds.keys()):
            param_state = comp.data_set(param, params[i], param_state)

        data_stimuli = comp.data_stimulate(current, None)

        return jnp.array(
            jx.integrate(
                comp,
                data_stimuli=data_stimuli,
                param_state=param_state,
                delta_t=dt,
                t_max=t_max,
            ).flatten()
        )

    def loss_fn(params, feat_loss_type: FeatureLoss):
        voltage_results = jax.vmap(simulate_network, in_axes=(None, 0))(params, amps)
        voltage_results = compress_traces(voltage_results, t_filter, dt, avg_scale)
        voltage_results = jnp.nan_to_num(voltage_results)

        normalized_v = normalize_trace_vmap(voltage_results)
        vf_mean, vf_logvar = jax.vmap(encoder)(normalized_v.astype(jnp.float32))

        if feat_loss_type == "kl":
            feat_loss = jnp.mean(
                0.5
                * jnp.sum(
                    ideal_logvar
                    - vf_logvar
                    + (
                        (jnp.exp(vf_logvar) + ((ideal_mean - vf_mean) ** 2))
                        / jnp.exp(ideal_logvar)
                    )
                    - 1,
                    axis=1,
                )
            )
        elif feat_loss_type == "mean":
            feat_loss = jnp.mean((ideal_mean - vf_mean) ** 2)
        elif feat_loss_type == "mse":
            feat_loss = jnp.mean((normalized_ideal_voltages - normalized_v) ** 2)
        else:
            raise TypeError(f"Invalid feature loss: {feat_loss_type}")

        return feat_loss

    ideal_voltages = jax.vmap(simulate_network, in_axes=(None, 0))(
        list(transformed_ideals.values()), amps
    )
    ideal_voltages = compress_traces(ideal_voltages, t_filter, dt, avg_scale)

    vae_config = get_vae_config(channel_type, vae_type, encoder_datestr)
    normalizer_type = vae_config["normalizer_type"]
    normalizer, _ = get_trace_normal_and_denormal(normalizer_type, channel_type)
    encoder = load_encoder(
        vae_type,
        channel_type,
        ideal_voltages.shape[1],
        encoder_datestr,
        epoch=encoder_epoch,
    )
    normalize_trace_vmap = jax.vmap(normalizer)
    normalized_ideal_voltages = normalize_trace_vmap(ideal_voltages)

    ideal_mean, ideal_logvar = jax.vmap(encoder)(
        normalized_ideal_voltages.astype(jnp.float32)
    )

    @jax.jit(static_argnums=(2,))
    def train_step(params, opt_state, optimizer):
        loss, grads = jax.value_and_grad(loss_fn)(params, feat_loss_type)
        updates, opt_state = optimizer.update(grads, opt_state, value=loss)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    optimal_list = list(transformed_ideals.values())
    optimal_array = jnp.array(optimal_list)
    distance_from_ideal = start_dist

    logger.info(f"ideal_params={ideal_params_dict}")

    if not new_distance_score:
        optimal_array_pos_prev = jnp.array(optimal_array)
        optimal_array_neg_prev = jnp.array(optimal_array)

        while True:
            # Increase the initial distance from the ideal
            optimal_array_pos = jnp.array(optimal_array + distance_from_ideal)
            optimal_array_neg = jnp.array(optimal_array - distance_from_ideal)

            logger.info(f"optimal_array_neg={optimal_array_neg}")
            logger.info(f"optimal_array_pos={optimal_array_pos}")

            # TODO: Optimal array has the proper init, train with it and see if it converges
            def get_post_training_optimal(opt):
                # optim = optax.chain(optax.zero_nans(), optax.polyak_sgd())
                if optimizer == optax.polyak_sgd:
                    optim = optimizer()
                elif optimizer == _jaxley_polyak:
                    optim = optimizer(1 / 3, 0.8)
                else:
                    optim = optimizer(learning_rate)
                opt_state = optim.init(opt)
                for step in range(
                    base_train_steps + int(train_steps_scaler * distance_from_ideal)
                ):
                    opt, opt_state, loss = train_step(opt, opt_state, optim)
                    loss = loss.item()
                    logger.info(
                        f"step={step}, loss={loss:.2e}, opt_params={opt}, ideal_params={optimal_array}"
                    )
                    if jnp.all(opt < optimal_array_pos_prev) and jnp.all(
                        opt > optimal_array_neg_prev
                    ):
                        logger.info(
                            "Reached within previous optimized bounds. Ending optimization."
                        )
                        return True

                return opt

            opt_pos_result = get_post_training_optimal(optimal_array_pos)
            opt_neg_result = get_post_training_optimal(optimal_array_neg)

            if (
                type(opt_pos_result) is not bool
                and jnp.max(jnp.abs(opt_pos_result - optimal_array)) > failure_distance
            ) or (
                type(opt_neg_result) is not bool
                and jnp.max(jnp.abs(opt_neg_result - optimal_array)) > failure_distance
            ):
                break

            optimal_array_pos_prev = optimal_array_pos
            optimal_array_neg_prev = optimal_array_neg

            distance_from_ideal += dist_stepsize

        return distance_from_ideal

    def get_best_post_training(init_seed):
        if optimizer == optax.polyak_sgd:
            optim = optimizer()
        elif optimizer == _jaxley_polyak:
            optim = optimizer(1 / 3, 0.8)
        else:
            optim = optimizer(learning_rate)

        init_key = jax.random.key(init_seed)
        opt_parm = jax.random.uniform(
            init_key,
            shape=(len(optimal_array),),
            minval=rand_init_bounds[0],
            maxval=rand_init_bounds[1],
        )
        opt_state = optim.init(opt_parm)
        best_loss = loss_fn(opt_parm, feat_loss_type)
        best_parms = jnp.array(opt_parm)

        for step in range(base_train_steps):
            opt_parm, opt_state, loss = train_step(opt_parm, opt_state, optim)
            loss = loss.item()

            if loss < best_loss:
                best_loss = loss
                best_parms = jnp.array(opt_parm)

            logger.info(
                f"step={step}, loss={loss:.2e}, opt_params={opt_parm}, best_parms={best_parms}, ideal_params={optimal_array}"
            )

        return best_parms

    basin = 0.15
    n_within_basin = 0
    for init_seed in INIT_SEEDS:
        best_params = get_best_post_training(init_seed)

        if jnp.linalg.norm(best_params - optimal_array) <= basin:
            logger.info(f"Seed {init_seed} success!")
            n_within_basin += 1
        else:
            logger.info(f"Seed {init_seed} failure!")

    return n_within_basin / len(INIT_SEEDS)


def _parse_args(config: dict) -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    for k, v in config.items():
        if type(v) is bool:
            parser.add_argument(
                f"--{k}",
                action=argparse.BooleanOptionalAction,
                default=v,
                help=f"Default: {f'--{k}' if v else f'--no-{k}'}",
            )
        else:
            parser.add_argument(
                f"--{k}",
                type=type(v) if v is not None else int,
                default=v,
                help=f"Default: {v}",
            )

    return parser.parse_args()


def main():
    with open("./configs/neuron_opt.yaml", "r") as f:
        config = yaml.safe_load(f)

    args = _parse_args(config)

    time_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_base = Path(f"./data/optimization_sweeps/{args.channel_type}")
    vae_type = MODEL_STRINGS[args.vae_type]

    dir_path = dir_base / time_str
    dir_path.mkdir(parents=True, exist_ok=True)

    if args.encoder_datestr == "latest":
        args.encoder_datestr = get_latest_datestr(args.channel_type, vae_type)

    logger = get_logger(dir_path / "log", name="generate_losses")

    log_commit(logger, "../")

    # Note: feat_loss can be MSE, which negates all encoder stuff
    opt_metric = _get_encoder_optimization_distance(
        args.channel_type,
        args.seed,
        vae_type,
        args.encoder_datestr,
        args.encoder_epoch,
        args.freeze_params,
        args.feat_loss,
        args.learning_rate,
        OPTIMIZERS[args.optimizer],
        args.start_dist,
        args.dist_stepsize,
        args.base_train_steps,
        args.train_steps_scaler,
        args.failure_distance,
        logger,
    )

    if args.feat_loss == "mse":
        args.encoder_datestr = None
        args.encoder_epoch = None

    args.opt_metric = opt_metric

    config_path = dir_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(vars(args), f, default_flow_style=False)

    logger.info(f"Final distance={opt_metric}")


if __name__ == "__main__":
    main()
