import argparse
import os
import sys
from itertools import batched
from typing import get_args

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from pathlib import Path

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from helper import (
    ChannelType,
    get_all_data,
    get_datagen_config,
)
from jax_feature_extractor import JaxEphysSweepFeatureExtractor
from logger import get_std_logger

from featuregrad.jaxley_ephys.allen_ephys.ephys_extractor import (
    EphysSweepFeatureExtractor,
)

jax.config.update("jax_enable_x64", True)


FEATS = [
    "adapt",
    "avg_rate",
    "first_isi",
    "isi_cv",
    "latency",
    "mean_isi",
    "median_isi",
]

TAKEN_FEATS = [
    # "avg_rate",
    "first_isi",
    # "isi_cv",
    # "latency",
    # "mean_isi",
    "median_isi",
]

MIN_SINGLE = ["adapt", "isi_cv"]
MAX_SINGLE = ["first_isi", "mean_isi", "median_isi"]
# MASK_IDX = jnp.array([1, 2, 4, 5])
MASK_IDX = jnp.array([0, 1])

NAN_FEATS = {k: np.nan for k in FEATS}


def _get_features(time_array, voltage_trace, current_trace):
    feature_ext = EphysSweepFeatureExtractor(time_array, voltage_trace, current_trace)
    feature_ext.process_spikes()
    features = feature_ext.as_dict()
    return {k: float(features[k]) for k in FEATS} if "adapt" in features else {}


def resave_allen(feat_path):
    feats_array = np.load(feat_path, allow_pickle=True)["arr_0"]
    feats_array[np.where(feats_array == {})[0]] = NAN_FEATS
    feats_dict = {k: np.array([feats[k] for feats in feats_array]) for k in FEATS}

    with open(feat_path, "wb") as f:
        np.savez(f, **feats_dict)


def get_spike_features(channel_type: ChannelType):
    datagen_config = get_datagen_config()
    dir_path = Path(datagen_config["save_path"])
    feat_file_path = dir_path / f"{channel_type}_feats.npz"

    spike_features = np.load(feat_file_path)
    return spike_features


def get_denan_and_scaled(spike_features: dict[str, np.ndarray]):
    d = {k: jnp.array(spike_features[k]) for k in TAKEN_FEATS}
    new_array = jnp.stack(list(d.values()), axis=-1)

    maxs = jnp.nanmax(new_array, axis=0)
    mins = jnp.nanmin(new_array, axis=0)

    mask = []
    for i, k in enumerate(TAKEN_FEATS):
        if k in MAX_SINGLE:
            mask.append(maxs[i])
        elif k in MIN_SINGLE:
            mask.append(mins[i])
    mask = jnp.array(mask)

    any_mask = jax.vmap(lambda x: jnp.any(jnp.isnan(x)))(new_array)
    all_mask = jax.vmap(lambda x: jnp.all(jnp.isnan(x)))(new_array)
    to_mask = jnp.logical_and(any_mask, jnp.logical_not(all_mask))

    def apply_mask(feats):
        if len(MASK_IDX) > 0:
            return feats.at[MASK_IDX].set(mask)
        else:
            return feats

    all_mask = jax.vmap(apply_mask)(new_array)
    new_no_nan = jnp.where(jnp.expand_dims(to_mask, 1), all_mask, new_array)
    new_scaled = (new_no_nan - mins) / (maxs - mins)

    return new_scaled, mask


def _save_denan_and_scaled(channel_type, feat_path):
    datagen_config = get_datagen_config()
    spike_features = np.load(feat_path)
    new_scaled, mask = get_denan_and_scaled(spike_features)

    dir_path = Path(datagen_config["save_path"])
    new_scaled_path = dir_path / f"{channel_type}_feats_norm.npz"
    mask_path = dir_path / f"{channel_type}_feats_mask.npz"

    with open(new_scaled_path, "wb") as f:
        np.savez(f, new_scaled)

    with open(mask_path, "wb") as f:
        np.savez(f, mask)


# This is the old and slow version of generating spike features. The JAX version
# below is much faster.
def _generate_spike_features_old(channel_type: ChannelType):
    logger = get_std_logger()

    datagen_config = get_datagen_config()
    data = get_all_data(channel_type, datagen_config)
    dt = datagen_config["dt"]
    t_max = datagen_config["t_max"]
    current_params = datagen_config["current"]
    t_filter = datagen_config["t_filter"]
    avg_scale = datagen_config["avg_scale"]

    voltages = data["voltages"]
    amps = data["amps"]

    time_array = np.array(
        jnp.arange(0, t_max - (2 * t_filter), (dt * avg_scale)) / 1000.0
    )
    currents = np.array(
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

    logger.info(f"Total traces to get features from: {len(voltages)}")
    feature_list = []
    for i in range(len(voltages)):
        feature_list.append(
            _get_features(
                time_array, np.array(voltages[i]), currents[i % len(currents)]
            )
        )
        if i % 200 == 0:
            logger.info(f"Processed {i} traces")

    feature_list = np.array(feature_list)

    dir_path = Path(datagen_config["save_path"])
    feat_file_path = dir_path / f"{channel_type}_feats-old.npz"
    with open(feat_file_path, "wb") as f:
        np.savez(f, feature_list)

    resave_allen(feat_file_path)


def _generate_spike_features_jax(channel_type: ChannelType):
    logger = get_std_logger()

    datagen_config = get_datagen_config()
    data = get_all_data(channel_type, datagen_config)
    dt = datagen_config["dt"]
    t_max = datagen_config["t_max"]
    t_filter = datagen_config["t_filter"]
    avg_scale = datagen_config["avg_scale"]

    voltages = data["voltages"]
    time_array = (
        jnp.arange(0, t_max - (2 * t_filter), (dt * avg_scale), jnp.float32) / 1000.0
    )
    max_spikes = 48

    extractor = JaxEphysSweepFeatureExtractor(time_array, max_spikes=max_spikes)
    extractor_jit = jax.jit(jax.vmap(extractor.features))
    batched_spikes = jax.jit(jax.vmap(extractor.process_spikes))

    batch_size = 8192
    feature_array = {k: jnp.zeros(len(voltages)) for k in FEATS}

    logger.info(f"Total traces to get features from: {len(voltages)}")
    i = 0
    for voltage_batch in batched(voltages, batch_size):
        voltage_batch = jnp.array(voltage_batch)
        batch_len = len(voltage_batch)
        feats = extractor_jit(voltage_batch)
        for feat in feats:
            feature_array[feat] = (
                feature_array[feat].at[i : i + batch_len].set(feats[feat])
            )
        overflow = bool(jnp.any(batched_spikes(voltage_batch).overflow))
        if overflow:
            raise RuntimeError(
                f"max_spikes={max_spikes} was too small for at least one trace; "
                "re-run with a larger --max-spikes"
            )
        i = i + batch_len
        logger.info(f"Processed {i} traces total.")

    dir_path = Path(datagen_config["save_path"])
    feat_file_path = dir_path / f"{channel_type}_feats.npz"
    with open(feat_file_path, "wb") as f:
        np.savez(f, **feature_array)

    _save_denan_and_scaled(channel_type, feat_file_path)


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Generate Ephys Features for Dataset")

    parser.add_argument(
        "data_type",
        type=str,
        help=f"Channel type from which to generate features: Possible values: {get_args(ChannelType)}",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    _generate_spike_features_jax(args.data_type)
    # _save_denan_and_scaled(args.data_type, "./data/voltage_traces/Pospischil_feats.npz")


if __name__ == "__main__":
    main()
