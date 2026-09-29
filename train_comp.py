import argparse
from datetime import datetime, timezone
from logging import DEBUG, Logger
from pathlib import Path
from typing import Type

import jax
import jax.numpy as jnp
import jaxley as jx
import optax
import yaml
from helper import (
    ChannelType,
    FeatureLoss,
    compress_traces,
    get_all_data,
    get_datagen_config,
    get_param_bounds,
    get_trace_normal_and_denormal,
    insert_channels,
    set_const_params,
)
from jax import Array
from jaxley import Compartment
from jaxley.optimize.transforms import SigmoidTransform
from logger import get_logger, log_commit, log_values
from numpy import array2string
from plot_simulation import plot_sweep
from vae.vae_base import VAE
from vae.vae_helper import (
    MODEL_STRINGS,
    get_latest_datestr,
    get_vae_config,
    load_encoder,
)

jax.config.update("jax_enable_x64", True)


def _get_transform_list(param_bounds: dict[str, Array]) -> list[SigmoidTransform]:
    transforms = []
    for param in param_bounds:
        transforms.append(
            SigmoidTransform(lower=param_bounds[param][0], upper=param_bounds[param][1])
        )
    return transforms


def _forward_transform_params(
    params: Array, transform_list: list[SigmoidTransform]
) -> Array:
    new_params = []
    assert len(params) == len(transform_list)
    for param, transform in zip(params, transform_list):
        new_params.append(transform.forward(param))

    return jnp.array(new_params)


def _backward_transform_params(
    params: Array, transform_list: list[SigmoidTransform]
) -> Array:
    new_params = []
    assert len(params) == len(transform_list)
    for param, transform in zip(params, transform_list):
        new_params.append(transform.inverse(param))

    return jnp.array(new_params)


def get_traces_to_match(
    channel_type: ChannelType, config: dict, paramset_idx: int
) -> Array:
    data = get_all_data(channel_type, config)
    voltages = data["voltages"]
    amps = data["amps"]
    params = data["params"]

    n_amps = len(amps)

    return (
        voltages[paramset_idx * n_amps : (paramset_idx + 1) * n_amps],
        amps,
        params[paramset_idx],
    )


def get_traces_to_match_rand(channel_type: ChannelType, config: dict, key) -> Array:
    data = get_all_data(channel_type, config)
    params = data["params"]

    idx = jax.random.randint(key, (), 0, len(params))

    return get_traces_to_match(channel_type, config, idx)


def _init_training_comp_params(
    comp: Compartment, param_bounds: dict, key: Array
) -> None:
    params = []
    for param in param_bounds:
        key, subkey = jax.random.split(key)
        val = jax.random.uniform(
            subkey, minval=param_bounds[param][0], maxval=param_bounds[param][1]
        )
        params.append(val)
        comp.make_trainable(param)

    return params


def set_frozen_params(
    param_bounds: dict, comp: Compartment, ideal_params_dict
) -> Compartment:
    for param in param_bounds:
        val = ideal_params_dict[param]
        comp.set(param, val)

    return comp


def _train_comp(
    channel_type: ChannelType,
    seed: int,
    steps: int,
    learning_rate: float,
    vae_type: Type[VAE],
    encoder_datestr: str,
    encoder_epoch: int | None,
    freeze_params: list[str],
    feat_loss: FeatureLoss,
    dir_path: Path,
    logger: Logger,
):
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

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    ideal_data = get_traces_to_match_rand(channel_type, config, subkey)

    ideal_voltages = ideal_data[0]
    amps = ideal_data[1]
    ideal_params = ideal_data[2]

    param_bounds = get_param_bounds(config["param_bounds"], channel_type)
    training_bounds = {k: v for k, v in param_bounds.items() if k not in freeze_params}
    freeze_bounds = {k: v for k, v in param_bounds.items() if k in freeze_params}
    ideal_params_dict = {p[0]: p[1] for p in zip(param_bounds.keys(), ideal_params)}

    comp = set_frozen_params(freeze_bounds, comp, ideal_params_dict)
    comp.init_states(delta_t=dt)

    params = _init_training_comp_params(comp, training_bounds, subkey)
    transform_list = _get_transform_list(training_bounds)
    opt_params = _backward_transform_params(params, transform_list)

    # Plot the params we want to train to
    plotting_currents = jnp.array(
        [
            jx.step_current(
                i_delay=current_params["delay"] - t_filter,
                i_dur=current_params["duration"],
                i_amp=amp,
                delta_t=(dt * avg_scale),
                t_max=t_max - 2 * t_filter - (dt * avg_scale),
            )
            for amp in amps
        ]
    )
    time_array = jnp.arange(0, t_max - (2 * t_filter), (dt * avg_scale))

    params_dict = {
        p[0]: array2string(p[1], formatter={"float": "{:.2e}".format})
        for p in zip(param_bounds.keys(), ideal_params)
    }
    params_dict = {k: v for k, v in params_dict.items() if k not in freeze_params}
    plot_sweep(
        ideal_voltages,
        plotting_currents,
        time_array,
        dir_path / "ideal_trace.pdf",
        params_dict,
    )

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

    ideal_mean, ideal_logvar = jax.vmap(encoder)(normalized_ideal_voltages)
    # logger.debug(f"ideal_mean={ideal_mean}, ideal_logvar={ideal_logvar}")

    if normalizer_type == "MinMax":
        bound_func = jax.vmap(lambda v: (jnp.min(v), jnp.max(v)))
    elif normalizer_type == "ZScore":
        bound_func = jax.vmap(lambda v: (jnp.mean(v), jnp.std(v)))
    # ZScoreGlobal keeps the absolute scale of the trace, so there is nothing
    # for a bounds loss to add back
    elif normalizer_type == "ZScoreGlobal":
        bound_func = jax.vmap(lambda v: (jnp.zeros(()), jnp.zeros(())))
    ideal_b1, ideal_b2 = bound_func(ideal_voltages)
    logger.debug(f"ideal_b1={ideal_b1}, ideal_b2={ideal_b2}")

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
        vb1, vb2 = bound_func(voltage_results)

        normalized_v = normalize_trace_vmap(voltage_results)
        vf_mean, vf_logvar = jax.vmap(encoder)(normalized_v)

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
            feat_loss = jnp.mean(jnp.sum(jnp.abs(ideal_mean - vf_mean), axis=1))
        elif feat_loss_type == "none":
            feat_loss = 0

        jax.debug.callback(log_values, logger, DEBUG, vb1=vb1, vb2=vb2)

        b1_loss = jnp.mean(jnp.abs(ideal_b1 - vb1))
        b2_loss = jnp.mean(jnp.abs(ideal_b2 - vb2))

        jax.debug.callback(
            log_values,
            logger,
            DEBUG,
            feat_loss=feat_loss,
            b1_loss=b1_loss,
            b2_loss=b2_loss,
        )

        return feat_loss + b1_loss + b2_loss

    @jax.jit(static_argnums=(2,))
    def train_step(params, opt_state, optimizer):
        loss, grads = jax.value_and_grad(loss_fn)(params, feat_loss)
        updates, opt_state = optimizer.update(grads, opt_state, value=loss)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    optim = optax.chain(optax.zero_nans(), optax.polyak_sgd())
    opt_state = optim.init(opt_params)

    logger.info(
        f"init_params={_forward_transform_params(opt_params, transform_list)}, ideal_params={ideal_params_dict}"
    )

    for step in range(steps):
        if step % 10 == 0:
            cur_voltages = jax.vmap(simulate_network, in_axes=(None, 0))(
                opt_params, amps
            )
            cur_voltages = compress_traces(cur_voltages, t_filter, dt, avg_scale)
            params_dict = {
                p[0]: array2string(p[1], formatter={"float": "{:.2e}".format})
                for p in zip(
                    training_bounds.keys(),
                    _forward_transform_params(opt_params, transform_list),
                )
            }
            plot_sweep(
                cur_voltages,
                plotting_currents,
                time_array,
                dir_path / f"plots/train_step{step:04}.pdf",
                params_dict,
            )

        opt_params, opt_state, loss = train_step(opt_params, opt_state, optim)
        loss = loss.item()
        forward_params = _forward_transform_params(opt_params, transform_list)
        logger.info(
            f"step={step}, loss={loss}, params={forward_params}, opt_params={opt_params}"
        )

    # Plot final result
    cur_voltages = jax.vmap(simulate_network, in_axes=(None, 0))(opt_params, amps)
    cur_voltages = compress_traces(cur_voltages, t_filter, dt, avg_scale)
    params_dict = {
        p[0]: array2string(p[1], formatter={"float": "{:.2e}".format})
        for p in zip(
            training_bounds.keys(),
            _forward_transform_params(opt_params, transform_list),
        )
    }
    plot_sweep(
        cur_voltages,
        plotting_currents,
        time_array,
        dir_path / f"plots/train_step{step:04}.pdf",
        params_dict,
    )

    return opt_params


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
    with open("./configs/comp.yaml", "r") as f:
        config = yaml.safe_load(f)

    args = _parse_args(config)

    channel_type = args.channel_type
    vae_type = MODEL_STRINGS[args.vae_type]

    if args.encoder_datestr == "latest":
        args.encoder_datestr = get_latest_datestr(args.channel_type, vae_type)

    time_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_base = Path(f"./data/models/comp/{channel_type}")

    dir_path = dir_base / time_str
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "plots").mkdir(parents=True, exist_ok=True)

    config_path = dir_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(vars(args), f, default_flow_style=False)

    logger = get_logger(dir_path / "log", name="train_comp")

    log_commit(logger, "../")

    _train_comp(
        channel_type,
        args.seed,
        args.steps,
        args.learning_rate,
        vae_type,
        args.encoder_datestr,
        args.encoder_epoch,
        args.freeze_params,
        args.feat_loss,
        dir_path,
        logger,
    )


if __name__ == "__main__":
    main()
