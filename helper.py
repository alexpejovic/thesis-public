import argparse
import glob
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, get_args

import jax.numpy as jnp
import numpy as np
import yaml
from jax import Array
from jaxley import Compartment
from jaxley.channels import HH
from jaxley.channels.pospischil import CaL, K, Km, Leak, Na

ChannelType = Literal["HH", "Pospischil", "Pospischil-bad"]
TraceDatasetKey = Literal[ChannelType, "Sines", "Pospischil-small"]
TraceNormalizer = Literal["MinMax", "ZScore", "MinMaxHard", "ZScoreGlobal"]
FeatureLoss = Literal["mean", "kl", "none"]

POSPISCHIL_PARAM_TO_PREFIX = {
    "gLeak": "Leak_",
    "eLeak": "Leak_",
    "gNa": "Na_",
    "gK": "K_",
    "gKm": "Km_",
    "taumax": "Km_",
    "gCaL": "CaL_",
    "eNa": "",
    "eK": "",
    "eCa": "",
    "vt": "",
}

HARDMIN = -90.0
HARDMAX = 50.0

# Traces per chunk when reducing a whole dataset off the memmap
GLOBAL_STATS_CHUNK = 4096


def normalize_trace_minmax(trace: Array) -> Array:
    trace = jnp.where(jnp.isnan(trace), 0.0, trace)
    mi = jnp.min(trace)
    ma = jnp.max(trace)
    denom = jnp.maximum(ma - mi, 1e-8)
    norm_trace = (2 * (trace - mi) / denom) - 1
    return norm_trace


def denormalize_trace_minmax(norm_trace: Array, mi: float, ma: float) -> Array:
    denom = jnp.maximum(ma - mi, 1e-8)
    denorm_trace = (denom / 2) * (norm_trace + 1) + mi
    return denorm_trace


def normalize_trace_zscore(trace: Array) -> Array:
    trace = jnp.where(jnp.isnan(trace), 0.0, trace)
    mean = jnp.mean(trace)
    std = jnp.std(trace)
    norm_trace = (trace - mean) / std
    # Below is done in case std is 0, which is rare so I just do this
    norm_trace = jnp.where(jnp.isnan(norm_trace), 0.0, norm_trace)
    return norm_trace


def denormalize_trace_zscore(norm_trace: Array, mean: float, std: float) -> Array:
    return (norm_trace * std) + mean


def normalize_trace_zscore_global(trace: Array, mean: float, std: float) -> Array:
    """ZScore against the mean/std of the whole dataset instead of the trace."""
    trace = jnp.where(jnp.isnan(trace), 0.0, trace)
    denom = jnp.maximum(std, 1e-8)
    norm_trace = (trace - mean) / denom
    return norm_trace


def denormalize_trace_zscore_global(
    norm_trace: Array, mean: float, std: float
) -> Array:
    return (norm_trace * jnp.maximum(std, 1e-8)) + mean


def normalize_trace_minmaxhard(trace: Array) -> Array:
    trace = jnp.where(jnp.isnan(trace), 0.0, trace)
    denom = jnp.maximum(HARDMAX - HARDMIN, 1e-8)
    norm_trace = (2 * (trace - HARDMIN) / denom) - 1
    return norm_trace


def denormalize_trace_minmaxhard(norm_trace: Array) -> Array:
    denom = jnp.maximum(HARDMAX - HARDMIN, 1e-8)
    denorm_trace = (denom / 2) * (norm_trace + 1) + HARDMIN
    return denorm_trace


def get_trace_normal_and_denormal(
    normalizer: TraceNormalizer,
    data_type: TraceDatasetKey | None = None,
    datagen_config: dict | None = None,
) -> tuple[Callable, Callable]:
    """Get the (normalize, denormalize) pair for a normalizer.

    `data_type` is only needed by dataset wide normalizers (ZScoreGlobal), which
    bake the stats of that dataset into the returned functions.
    """
    if normalizer == "MinMax":
        return normalize_trace_minmax, denormalize_trace_minmax
    elif normalizer == "ZScore":
        return normalize_trace_zscore, denormalize_trace_zscore
    elif normalizer == "MinMaxHard":
        return normalize_trace_minmaxhard, denormalize_trace_minmaxhard
    elif normalizer == "ZScoreGlobal":
        if data_type is None:
            raise TypeError("ZScoreGlobal needs a data_type to take its stats from")
        mean, std = get_global_trace_stats(data_type, datagen_config)
        return (
            partial(normalize_trace_zscore_global, mean=mean, std=std),
            partial(denormalize_trace_zscore_global, mean=mean, std=std),
        )
    else:
        raise TypeError("Invalid Normalizer")


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "channel_type",
        type=str,
        help=f"Type of the cell: Possible values: {get_args(ChannelType)}",
    )
    return parser.parse_args()


def get_param_bounds(param_bounds: dict, channel_type: ChannelType) -> dict[str, list]:
    get_key = _get_key_factory(channel_type)

    if channel_type in ["HH", "Pospischil"]:
        return {
            get_key(key): param_bounds[channel_type][key]
            for key in param_bounds[channel_type]
        }
    elif channel_type == "Pospischil-bad":
        pospischil_bounds = {
            get_key(key): param_bounds["Pospischil"][key]
            for key in param_bounds["Pospischil"]
        }
        bad_bounds = {
            get_key(key): param_bounds[channel_type][key]
            for key in param_bounds[channel_type]
        }
        for k, v in bad_bounds.items():
            pospischil_bounds[k] = v

        return pospischil_bounds


def set_const_params(
    param_consts: dict, channel_type: ChannelType, comp: Compartment
) -> Compartment:
    params = param_consts[channel_type]
    get_key = _get_key_factory(channel_type)

    for param in params:
        comp.set(get_key(param), params[param])

    return comp


def _get_key_factory(channel_type: ChannelType) -> Callable[[str], str]:
    if channel_type == "HH":
        get_key = lambda key: f"{channel_type}_{key}"
    elif channel_type == "Pospischil" or channel_type == "Pospischil-bad":
        get_key = lambda key: f"{POSPISCHIL_PARAM_TO_PREFIX[key]}{key}"
    else:
        raise TypeError("Very Bad!")
    return get_key


def insert_channels(comp: Compartment, channel_type: ChannelType) -> Compartment:
    if channel_type == "HH":
        comp.insert(HH())
    elif channel_type == "Pospischil" or channel_type == "Pospischil-bad":
        comp.insert(Leak())
        comp.insert(Na())
        comp.insert(K())
        comp.insert(Km())
        comp.insert(CaL())
    else:
        raise TypeError("Very Bad!")

    return comp


def get_datagen_config():
    with open("./configs/datagen.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def get_vae_config():
    with open("./configs/vae.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def get_beta_mse_config():
    with open("./configs/beta_mse.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def get_region_scores_config():
    with open("./configs/region_scores.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def get_convexity_vs_opt_config():
    with open("./configs/convexity_vs_opt.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def get_voltage_traces(data_type: TraceDatasetKey, datagen_config: dict | None = None):
    if datagen_config == None:
        datagen_config = get_datagen_config()
    dt = datagen_config["dt"]
    t_max = datagen_config["t_max"]
    t_filter = datagen_config["t_filter"]
    avg_scale = datagen_config["avg_scale"]
    y_axis = int((t_max - 2 * t_filter) / (dt * avg_scale))

    dir_path = Path(datagen_config["save_path"])

    voltages = np.memmap(
        dir_path / f"{data_type}.dat",
        dtype="float32",
        mode="r",
    ).reshape((-1, y_axis))

    return voltages


def get_global_trace_stats(
    data_type: TraceDatasetKey, datagen_config: dict | None = None
) -> tuple[float, float]:
    """Mean and std over every sample of every trace of a dataset.

    The datasets are far too big to reduce on every run, so the result is
    cached next to the traces. The cache is keyed on the sample count, so
    regenerating a dataset with a different size invalidates it.
    """
    if datagen_config == None:
        datagen_config = get_datagen_config()

    voltages = get_voltage_traces(data_type, datagen_config)
    stats_path = Path(datagen_config["save_path"]) / f"{data_type}_global_stats.npz"

    if stats_path.exists():
        stats = np.load(stats_path)
        if int(stats["n_samples"]) == voltages.size:
            return float(stats["mean"]), float(stats["std"])

    # Streamed in chunks, the memmap does not fit in memory
    total = 0.0
    total_sq = 0.0
    for start in range(0, voltages.shape[0], GLOBAL_STATS_CHUNK):
        chunk = np.asarray(
            voltages[start : start + GLOBAL_STATS_CHUNK], dtype=np.float64
        )
        # Same nan handling as the per trace normalizers
        chunk = np.where(np.isnan(chunk), 0.0, chunk)
        total += chunk.sum()
        total_sq += np.square(chunk).sum()

    mean = total / voltages.size
    std = np.sqrt(max((total_sq / voltages.size) - mean**2, 0.0))

    np.savez(stats_path, mean=mean, std=std, n_samples=voltages.size)

    return float(mean), float(std)


def unit(y: np.ndarray) -> np.ndarray:
    """Min-max a loss landscape onto [0, 1].

    A NaN means the simulation blew up at that grid point, so it is pinned to the
    worst finite loss rather than dropped -- dropping would change the length of
    one of the two arrays being compared.
    """
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    if not finite.all():
        if not finite.any():
            raise RuntimeError("landscape is entirely non-finite")
        y = np.where(finite, y, y[finite].max())
    lo, hi = float(y.min()), float(y.max())
    if hi <= lo:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def unit_robust(y: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Like `unit`, but anchored on percentiles instead of the true min/max.

    `unit` divides by the global min/max, so one extreme grid point rescales the
    whole landscape. Measured over 56 subsets that cost the beta=0 encoder its
    win: it is the most MSE-like landscape by rank agreement (55/56) yet has the
    WIDEST min-max distance spread of any beta (sd 0.011 vs 0.003-0.007), because
    it is being charged for a different normalization anchor rather than a
    different shape. Clipping to [0, 1] after a percentile anchor keeps the
    original formula while making it robust to that.

    Defaults match `configs/beta_mse.yaml:unit_robust_pct`; callers driven by that
    config should pass the configured values rather than rely on these.
    """
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    if not finite.all():
        if not finite.any():
            raise RuntimeError("landscape is entirely non-finite")
        y = np.where(finite, y, y[finite].max())
    lo, hi = np.percentile(y, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(y)
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)


def compress_traces(traces, t_filter, dt, avg_scale):
    # Filter out first and last 10 ms
    traces = traces[:, int(t_filter / dt) : -int(t_filter / dt) - 1]

    # Average trace
    traces = jnp.mean(traces.reshape(traces.shape[0], -1, avg_scale), axis=2)

    return traces


def get_all_data(
    channel_type: ChannelType, datagen_config: dict | None = None
) -> dict[str, Any]:
    if datagen_config == None:
        datagen_config = get_datagen_config()

    data_dir = Path(datagen_config["save_path"])
    voltages = get_voltage_traces(channel_type, datagen_config)
    amps = jnp.load(data_dir / f"{channel_type}_amps.npz")["arr_0"]
    params = jnp.load(data_dir / f"{channel_type}_params.npz")["arr_0"]

    return {"voltages": voltages, "amps": amps, "params": params}


def filter_configs(config_file_list: list[str], line: str):
    """Return a list back of all files who contain line"""
    grep_sb = subprocess.run(
        [
            "grep",
            "-l",
            line,
            *config_file_list,
        ],
        capture_output=True,
        text=True,
    )
    filename_list = grep_sb.stdout.split("\n")
    filename_list.remove("")
    filename_list = list(filter(lambda s: "latest" not in s, filename_list))

    return filename_list


def get_feature_losses(
    channel_type,
    encoder_date,
    target_freeze_params,
    named_encoder=False,
    feat_loss: str = "mean",
):
    if named_encoder:
        try:
            encoder_date = (
                next(Path("./data/models/vae/named/").glob(encoder_date))
                .readlink()
                .name
            )
        except StopIteration:
            print("Named encoder does not exist", file=sys.stderr)
            exit(-1)
        except OSError:
            print("Could not read named encoder symlink", file=sys.stderr)
            exit(-1)

    filename_list = glob.glob(f"data/comp_losses/{channel_type}/*/*.yaml")
    to_grep = [f"encoder_datestr: '{encoder_date}'", f"feat_loss: {feat_loss}"]

    for line in to_grep:
        filename_list = filter_configs(filename_list, line)

    if len(filename_list) == 0:
        raise KeyError(f"Couldn't fine sweeps with lines {str(to_grep)}")

    freeze_param_list = []

    for config_name in filename_list:
        with open(config_name, "r") as f:
            freeze_param_list.append(yaml.safe_load(f)["freeze_params"])
    valid_sweeps = list(
        filter(
            lambda li: li[1] == target_freeze_params,
            enumerate(freeze_param_list),
        )
    )

    if len(valid_sweeps) == 0:
        print(filename_list)
        print(freeze_param_list)
        print(target_freeze_params)
        raise KeyError(
            "Feature loss sweep with the desired freeze parameters not found."
        )

    # The -1 indexing here might be a bug: (we want the most recent config)
    sweep_config_file = Path(filename_list[valid_sweeps[-1][0]])
    sweep_dir = sweep_config_file.parent
    with open(sweep_config_file, "r") as f:
        params_tuple = list(yaml.safe_load(f)["params_dict"].items())

    feat_losses = np.load(sweep_dir / "feat_losses.npz")["arr_0"]
    opt_samples = np.load(sweep_dir / "opt_samples.npz")["arr_0"]
    mses = np.load(sweep_dir / "mses.npz")["arr_0"]
    try:
        jaxley_losses = np.load(sweep_dir / "jaxley_losses.npz")["arr_0"]
    except FileNotFoundError:
        jaxley_losses = None
    try:
        allen_losses = np.load(sweep_dir / "allen_losses.npz")["arr_0"]
    except FileNotFoundError:
        allen_losses = None

    return feat_losses, opt_samples, params_tuple, mses, allen_losses, jaxley_losses


def get_opt_sweep_score(
    channel_type,
    encoder_date,
    target_freeze_params,
    named_encoder=False,
    target_optimizer="polyak",
    target_feat_loss="mean",
):
    if named_encoder:
        try:
            encoder_date = (
                next(Path("./data/models/vae/named/").glob(encoder_date))
                .readlink()
                .name
            )
        except StopIteration:
            print("Named encoder does not exist", file=sys.stderr)
            exit(-1)
        except OSError:
            print("Could not read named encoder symlink", file=sys.stderr)
            exit(-1)

    filename_list = glob.glob(
        f"./data/optimization_sweeps/{channel_type}/*/config.yaml"
    )

    if encoder_date == None:
        enc_date_grep_line = "encoder_datestr: null"
        assert target_feat_loss == "mse"
    else:
        enc_date_grep_line = f"encoder_datestr: '{encoder_date}'"

    to_grep = [enc_date_grep_line, f"feat_loss: {target_feat_loss}"]
    for line in to_grep:
        filename_list = filter_configs(filename_list, line)

    freeze_params_and_opts = []

    for config_name in filename_list:
        with open(config_name, "r") as f:
            d = yaml.safe_load(f)
            freeze_params_and_opts.append((d["freeze_params"], d["optimizer"]))
    valid_sweeps = list(
        filter(
            lambda item: item[1][0] == target_freeze_params
            and item[1][1] == target_optimizer,
            enumerate(freeze_params_and_opts),
        )
    )
    # The -1 indexing here might be a bug: (we want the most recent config)
    sweep_config_file = Path(filename_list[valid_sweeps[-1][0]])
    with open(sweep_config_file, "r") as f:
        opt_metric = yaml.safe_load(f)["opt_metric"]

    return opt_metric
