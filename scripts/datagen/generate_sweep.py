import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)


from itertools import product
from logging import Logger
from typing import Type

import jax
import jaxley as jx
import yaml
from helper import (
    ChannelType,
    FeatureLoss,
    get_datagen_config,
    get_param_bounds,
    insert_channels,
    set_const_params,
)
from jaxley import Compartment
from logger import get_logger, log_commit
from train_comp import (
    _backward_transform_params,
    _forward_transform_params,
    _get_transform_list,
    compress_traces,
    get_traces_to_match_rand,
    set_frozen_params,
)
from vae.vae_base import VAE
from vae.vae_helper import (
    MODEL_STRINGS,
    get_latest_datestr,
)

jax.config.update("jax_enable_x64", True)


def _get_small_sweep(
    channel_type: ChannelType,
    opt_bounds: tuple[float, float],
    opt_samples: int,
    seed: int,
    freeze_params: list[str],
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

    amps = ideal_data[1]

    param_bounds = get_param_bounds(config["param_bounds"], channel_type)
    training_bounds = {k: v for k, v in param_bounds.items() if k not in freeze_params}
    freeze_bounds = {k: v for k, v in param_bounds.items() if k in freeze_params}

    ideal_params_dict = {}
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

    ideal_voltages = jax.vmap(simulate_network, in_axes=(None, 0))(
        list(transformed_ideals.values()), amps
    )
    ideal_voltages = compress_traces(ideal_voltages, t_filter, dt, avg_scale)

    opt_sample_length = (opt_bounds[1] - opt_bounds[0]) / (opt_samples - 1)
    n_params = len(training_bounds)
    opt_single_iterator = lambda: (
        x * opt_sample_length + opt_bounds[0] for x in range(opt_samples)
    )
    opt_iterator = product(*[opt_single_iterator() for _ in range(n_params)])
    opt_sample_list = [
        x * opt_sample_length + opt_bounds[0] for x in range(opt_samples)
    ]

    v_res_list = []
    for i, opt_param_sample in enumerate(opt_iterator):
        voltage_results = jax.vmap(simulate_network, in_axes=(None, 0))(
            jnp.array(opt_param_sample), amps
        )
        v_res_list.append(voltage_results)
        logger.info(f"Completed {i + 1} sweep calculations")

    return jnp.array(v_res_list), opt_sample_list, ideal_voltages


def _parse_args(config: dict) -> argparse.Namespace:
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
    with open("./configs/loss_sweep.yaml", "r") as f:
        config = yaml.safe_load(f)

    args = _parse_args(config)

    time_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_base = Path(f"./data/comp_small_sweeps/{args.channel_type}")
    vae_type = MODEL_STRINGS[args.vae_type]

    dir_path = dir_base / time_str
    dir_path.mkdir(parents=True, exist_ok=True)

    if args.encoder_datestr == "latest":
        args.encoder_datestr = get_latest_datestr(args.channel_type, vae_type)

    config_path = dir_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(vars(args), f, default_flow_style=False)

    logger = get_logger(dir_path / "log", name="generate_losses")

    log_commit(logger, "../")

    voltages, opt_samples, ideal_voltages = _get_small_sweep(
        args.channel_type,
        args.opt_bounds,
        args.opt_samples,
        args.seed,
        args.freeze_params,
        logger,
    )

    with open(dir_path / "voltages.npz", "wb") as f:
        jnp.savez(f, jnp.array(voltages))

    with open(dir_path / "opt_samples.npz", "wb") as f:
        jnp.savez(f, jnp.array(opt_samples))

    with open(dir_path / "ideal_voltages.npz", "wb") as f:
        jnp.savez(f, jnp.array(ideal_voltages))

    # Create a simlink to the newest
    link_path = Path(dir_base / "latest")
    if link_path.exists() and link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(time_str, target_is_directory=True)


if __name__ == "__main__":
    main()
