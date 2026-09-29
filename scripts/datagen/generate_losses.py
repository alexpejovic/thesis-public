import argparse
import os
import sys
from datetime import datetime, timezone
from functools import partial
from itertools import batched, product
from logging import INFO, Logger
from pathlib import Path
from typing import Type

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import yaml
from jaxley import Compartment

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)


from allen_feats import MASK_IDX, TAKEN_FEATS
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
from jax_feature_extractor import JaxEphysSweepFeatureExtractor
from logger import get_logger, log_commit, log_values
from losses import _make_jaxley_paper_loss
from train_comp import (
    _backward_transform_params,
    _forward_transform_params,
    _get_transform_list,
    get_traces_to_match_rand,
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
    parser.add_argument(
        "--calc_allen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default: --no-calc_allen",
    )

    return parser.parse_args()


def _get_allen_extractor_fn(channel_type, time_array, max_spikes):
    extractor = JaxEphysSweepFeatureExtractor(time_array, max_spikes=max_spikes)
    extractor = extractor.features

    datagen_config = get_datagen_config()
    dir_path = Path(datagen_config["save_path"])
    mask_path = dir_path / f"{channel_type}_feats_mask.npz"
    mask = jnp.load(mask_path)["arr_0"]
    check = lambda f: jnp.logical_and(
        jnp.any(jnp.isnan(f)), jnp.logical_not(jnp.all(jnp.isnan(f)))
    )

    def _apply_mask(feats):
        if len(MASK_IDX) > 0:
            return feats.at[MASK_IDX].set(mask)
        else:
            return feats

    mask_feat = lambda f: jnp.where(check(f), _apply_mask(f), f)

    feats_path = dir_path / f"{channel_type}_feats.npz"
    all_feats = np.load(feats_path)

    d = {k: jnp.array(all_feats[k]) for k in TAKEN_FEATS}
    new_array = jnp.stack(list(d.values()), axis=-1)

    maxs = jnp.nanmax(new_array, axis=0)
    mins = jnp.nanmin(new_array, axis=0)

    def _new_ext(voltage_trace):
        features = extractor(voltage_trace)
        new_thing = jnp.zeros(len(TAKEN_FEATS))
        for i, feat in enumerate(TAKEN_FEATS):
            new_thing = new_thing.at[i].set(features[feat])

        new_thing = mask_feat(new_thing)
        new_thing = (new_thing - mins) / (maxs - mins)

        return new_thing

    return _new_ext


def _get_sweep_losses(
    channel_type: ChannelType,
    opt_bounds: tuple[float, float],
    opt_samples: int,
    seed: int,
    vae_type: Type[VAE],
    encoder_datestr: str,
    encoder_epoch: int | None,
    freeze_params: list[str],
    feat_loss_type: FeatureLoss,
    calc_allen: bool,
    logger: Logger,
):
    comp = Compartment()

    comp = insert_channels(comp, channel_type)

    datagen_cfg = get_datagen_config()
    dt = datagen_cfg["dt"]
    t_max = datagen_cfg["t_max"]
    current_params = datagen_cfg["current"]
    t_filter = datagen_cfg["t_filter"]
    avg_scale = datagen_cfg["avg_scale"]

    time_array = (
        jnp.arange(0, t_max - (2 * t_filter), (dt * avg_scale), jnp.float32) / 1000.0
    )
    max_spikes = 48

    extract_allen_feats = jax.jit(
        jax.vmap(_get_allen_extractor_fn(channel_type, time_array, max_spikes))
    )

    comp = set_const_params(datagen_cfg["param_consts"], channel_type, comp)
    comp.record("v")

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    amps = get_traces_to_match_rand(channel_type, datagen_cfg, subkey)[1]

    param_bounds = get_param_bounds(datagen_cfg["param_bounds"], channel_type)
    training_bounds = {k: v for k, v in param_bounds.items() if k not in freeze_params}
    freeze_bounds = {k: v for k, v in param_bounds.items() if k in freeze_params}
    transform_list = _get_transform_list(training_bounds)

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
    ideal_allen_feats = extract_allen_feats(ideal_voltages)

    if jnp.any(jnp.isnan(ideal_allen_feats)):
        raise RuntimeError("Ideal trace is non-spiking")

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
    normalized_ideal_voltages = normalize_trace_vmap(ideal_voltages.astype(jnp.float32))

    ideal_mean, ideal_logvar = jax.vmap(encoder)(normalized_ideal_voltages)

    if normalizer_type == "MinMax":
        bound_func = jax.vmap(lambda v: (jnp.min(v), jnp.max(v)))
    elif normalizer_type == "ZScore":
        bound_func = jax.vmap(lambda v: (jnp.mean(v), jnp.std(v)))
    # MinMaxHard and ZScoreGlobal mean that the bounds exist within encoding, so
    # there should be no bounds loss for these
    elif normalizer_type == "MinMaxHard" or normalizer_type == "ZScoreGlobal":
        bound_func = jax.vmap(lambda _: (jnp.zeros(()), jnp.zeros(())))
    ideal_b1, ideal_b2 = bound_func(ideal_voltages)
    logger.debug(f"ideal_b1={ideal_b1}, ideal_b2={ideal_b2}")

    jaxley_loss_f = _make_jaxley_paper_loss("full", channel_type, datagen_cfg)

    @jax.jit(static_argnames=["feat_loss_type", "calc_allen"])
    @partial(jax.vmap, in_axes=(0, None, None))
    def loss_fn(params, feat_loss_type: FeatureLoss, calc_allen: bool):
        voltage_results = jax.vmap(simulate_network, in_axes=(None, 0))(params, amps)
        voltage_results = compress_traces(voltage_results, t_filter, dt, avg_scale)
        voltage_results = jnp.nan_to_num(voltage_results)

        vb1, vb2 = bound_func(voltage_results)

        normalized_v = normalize_trace_vmap(voltage_results)
        vf_mean, vf_logvar = jax.vmap(encoder)(normalized_v.astype(jnp.float32))

        mse = jnp.mean((normalized_ideal_voltages - normalized_v) ** 2)

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
        elif feat_loss_type == "none":
            feat_loss = 0

        b1_loss = jnp.mean(jnp.abs(ideal_b1 - vb1))
        b2_loss = jnp.mean(jnp.abs(ideal_b2 - vb2))

        jaxley_loss = jaxley_loss_f(ideal_voltages, voltage_results)
        # jax.debug.callback(
        #     log_values,
        #     logger,
        #     INFO,
        #     jaxley_loss=jaxley_loss,
        #     jl_shape=jaxley_loss.shape,
        #     feat_loss=feat_loss,
        #     fl_shape=feat_loss.shape,
        # )

        if calc_allen:
            allen_feats = extract_allen_feats(voltage_results)
            allen_loss = jnp.mean((allen_feats - ideal_allen_feats) ** 2)
            allen_loss = jnp.where(jnp.any(jnp.isnan(allen_feats)), mse, allen_loss)

            return feat_loss + b1_loss + b2_loss, (
                b1_loss + b2_loss,
                feat_loss,
                mse,
                allen_loss,
                jaxley_loss,
            )
        else:
            return feat_loss + b1_loss + b2_loss, (
                b1_loss + b2_loss,
                feat_loss,
                mse,
                None,
                jaxley_loss,
            )

    opt_sample_length = (opt_bounds[1] - opt_bounds[0]) / (opt_samples - 1)
    n_params = len(training_bounds)
    opt_single_iterator = lambda: (
        x * opt_sample_length + opt_bounds[0] for x in range(opt_samples)
    )
    opt_iterator = product(*[opt_single_iterator() for _ in range(n_params)])
    opt_sample_list = [
        x * opt_sample_length + opt_bounds[0] for x in range(opt_samples)
    ]

    losses = []
    feat_losses = []
    b_losses = []
    mses = []
    allen_losses = []
    jaxley_losses = []
    batch_size = 512
    for i, opt_param_sample in enumerate(batched(opt_iterator, batch_size)):
        loss, (b_loss, feat_loss, mse, allen_loss, jaxley_loss) = loss_fn(
            jnp.array(opt_param_sample), feat_loss_type, calc_allen
        )
        losses.append(loss)
        feat_losses.append(feat_loss)
        b_losses.append(b_loss)
        mses.append(mse)
        allen_losses.append(allen_loss)
        jaxley_losses.append(jaxley_loss)
        logger.info(f"Completed {i * batch_size} loss calculations")

    losses = jnp.concatenate(losses).flatten()
    feat_losses = jnp.concatenate(feat_losses).flatten()
    b_losses = jnp.concatenate(b_losses).flatten()
    mses = jnp.concatenate(mses).flatten()
    allen_losses = jnp.concatenate(allen_losses).flatten() if calc_allen else None
    jaxley_losses = jnp.concatenate(jaxley_losses).flatten()
    return (
        losses,
        (b_losses, feat_losses, mses, allen_losses, jaxley_losses),
        opt_sample_list,
        transformed_ideals,
    )


def main():
    with open("./configs/loss_sweep.yaml", "r") as f:
        config = yaml.safe_load(f)

    args = _parse_args(config)

    time_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
    dir_base = Path(f"./data/comp_losses/{args.channel_type}")
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

    (
        losses,
        (b_losses, feat_losses, mses, allen_losses, jaxley_losses),
        opt_samples,
        params_dict,
    ) = _get_sweep_losses(
        args.channel_type,
        args.opt_bounds,
        args.opt_samples,
        args.seed,
        vae_type,
        args.encoder_datestr,
        args.encoder_epoch,
        args.freeze_params,
        args.feat_loss,
        args.calc_allen,
        logger,
    )

    args.params_dict = params_dict
    with open(config_path, "w") as f:
        yaml.safe_dump(vars(args), f, default_flow_style=False)

    with open(dir_path / "losses.npz", "wb") as f:
        jnp.savez(f, jnp.array(losses))

    with open(dir_path / "b_losses.npz", "wb") as f:
        jnp.savez(f, jnp.array(b_losses))

    with open(dir_path / "feat_losses.npz", "wb") as f:
        jnp.savez(f, jnp.array(feat_losses))

    with open(dir_path / "opt_samples.npz", "wb") as f:
        jnp.savez(f, jnp.array(opt_samples))

    with open(dir_path / "mses.npz", "wb") as f:
        jnp.savez(f, jnp.array(mses))

    with open(dir_path / "jaxley_losses.npz", "wb") as f:
        jnp.savez(f, jnp.array(jaxley_losses))

    if args.calc_allen:
        with open(dir_path / "allen_losses.npz", "wb") as f:
            jnp.savez(f, jnp.array(allen_losses))

    # Create a simlink to the newest
    link_path = Path(dir_base / "latest")
    if link_path.exists() and link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(time_str, target_is_directory=True)


if __name__ == "__main__":
    main()
