import argparse
from pathlib import Path
from typing import Type, get_args

import jax
import jax.numpy as jnp
import jaxley as jx
import matplotlib.pyplot as plt
import numpy as np
import yaml
from helper import (
    ChannelType,
    TraceDatasetKey,
    TraceNormalizer,
    get_all_data,
    get_datagen_config,
    get_param_bounds,
    get_trace_normal_and_denormal,
    get_voltage_traces,
)
from jax import Array
from vae.vae_helper import VAE, get_available_vae_models, get_latest_datestr, load_vae

CMAP = plt.get_cmap("Spectral")


def get_compressed_currents(channel_type, datagen_config=None):
    if datagen_config is None:
        datagen_config = get_datagen_config()

    dt = datagen_config["dt"]
    t_max = datagen_config["t_max"]
    t_filter = datagen_config["t_filter"]
    avg_scale = datagen_config["avg_scale"]
    current_params = datagen_config["current"]

    data = get_all_data(channel_type, datagen_config)
    amps = data["amps"]

    currents = jnp.array(
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
    return currents


def get_compressed_time_array(channel_type, datagen_config=None):
    if datagen_config is None:
        datagen_config = get_datagen_config()

    dt = datagen_config["dt"]
    t_max = datagen_config["t_max"]
    t_filter = datagen_config["t_filter"]
    avg_scale = datagen_config["avg_scale"]

    time_array = jnp.arange(0, t_max - (2 * t_filter), (dt * avg_scale))

    return time_array


def _gen_plots(channel_type: ChannelType):
    config = get_datagen_config()
    param_bounds = get_param_bounds(config["param_bounds"], channel_type)

    data = get_all_data(channel_type, config)
    voltages = data["voltages"]
    amps = data["amps"]
    params = data["params"]

    n_amps = len(amps)

    b = len(voltages) // n_amps
    assert b == len(params)
    n_to_plot = 100
    time_array = get_compressed_time_array(channel_type, config)

    currents = get_compressed_currents(channel_type, config)

    dir_path = Path(f"./plots/{channel_type}")
    dir_path.mkdir(parents=True, exist_ok=True)

    if b > n_to_plot:
        param_sweep_idxs = jax.tree_util.tree_map(
            jnp.round, jnp.linspace(0, b, n_to_plot, endpoint=False)
        ).astype(int)
    else:
        param_sweep_idxs = jnp.arange(b)

    # key = jax.random.PRNGKey(1000)
    for i, param_sweep_idx in enumerate(param_sweep_idxs):
        # Uncomment below to randomly index the voltage traces instead
        # key, subkey = jax.random.split(key)
        # param_sweep_idx = jax.random.randint(subkey, (), 0, b)
        voltage = voltages[param_sweep_idx * n_amps : (param_sweep_idx + 1) * n_amps]
        params_dict = {
            p[0]: np.array2string(p[1], formatter={"float": "{:.2e}".format})
            for p in zip(param_bounds.keys(), params[param_sweep_idx])
        }

        plot_sweep(
            voltage,
            currents,
            time_array,
            dir_path / f"{i:02}.pdf",
            params_dict=params_dict,
        )


def plot_sweep(voltage_traces, currents, time_array, save_path, params_dict=None):
    colors = CMAP(jnp.linspace(0, 1, len(currents)))

    if params_dict is None:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [6, 2, 1]}
        )

    for j in range(len(currents)):
        ax1.plot(time_array, voltage_traces[j], color=colors[j])

    ax1.sharex(ax2)
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Voltage and Current Traces")
    for j in range(len(currents)):
        ax2.plot(time_array, currents[j] * 1000.0, color=colors[j])
    ax2.set_ylabel("Current (pA)")
    ax2.set_xlabel("Time (ms)")

    if params_dict is not None:
        ax3.axis("off")
        table = plt.table(
            cellText=[list(params_dict.values())],
            colLabels=list(params_dict.keys()),
            loc="center",
        )
        table.auto_set_font_size(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_trace_and_norm(
    voltage_trace,
    dt,
    normalizer_type: TraceNormalizer,
    data_type: TraceDatasetKey | None = None,
):
    normalizer, _ = get_trace_normal_and_denormal(normalizer_type, data_type)
    voltage_trace_n = normalizer(voltage_trace)
    time_array = jnp.arange(0, voltage_trace_n.shape[0]) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.set_title(f"Voltage Trace and Normalizer {normalizer_type}")
    ax1.set_ylabel("Voltage")
    ax1.set_xlabel("Time (ms)")
    ax1.plot(time_array, voltage_trace)
    ax2.plot(time_array, voltage_trace_n)
    ax2.set_ylabel("Normalized Voltage")
    ax2.tick_params(labelbottom=False)
    ax1.sharex(ax2)

    fig.tight_layout()
    # fig.savefig(save_path)
    # plt.close(fig)
    plt.show()


def plot_trace_and_ft(
    voltage_trace,
    dt,
    normalizer_type: TraceNormalizer,
    data_type: TraceDatasetKey | None = None,
):
    normalizer, _ = get_trace_normal_and_denormal(normalizer_type, data_type)
    voltage_trace_n = normalizer(voltage_trace)
    time_array = jnp.arange(0, voltage_trace_n.shape[0]) * dt
    ft = jnp.fft.rfft(voltage_trace_n)
    freq = jnp.fft.rfftfreq(voltage_trace_n.shape[0], d=dt)
    magnitude = jnp.sqrt(ft.real**2 + ft.imag**2)
    phase = jnp.atan2(ft.imag, ft.real)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [5, 2, 2]}
    )

    ax1.plot(time_array, voltage_trace_n)

    ax1.set_ylabel("Normalized Voltage")
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Normalized Voltage Trace and Fourier Transform")
    ax2.plot(freq, magnitude)
    ax2.set_ylabel("Magnitude")
    ax3.plot(freq, phase)
    ax3.set_ylabel("Phase")
    ax3.set_xlabel("Freq (kHz)")
    ax2.sharex(ax3)
    ax2.tick_params(labelbottom=False)

    fig.tight_layout()
    # fig.savefig(save_path)
    # plt.close(fig)
    plt.show()


def _plot_trace_basic(trace: Array, dt: int, save_path: Path):
    time_array = jnp.arange(0, trace.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title("Basic Voltage Trace")
    ax.set_ylabel("Voltage")
    ax.set_xlabel("Time (ms)")
    ax.plot(time_array, trace, label="Original Trace")

    fig.tight_layout()
    fig.legend()
    fig.savefig(save_path)
    plt.close(fig)


def plot_traces_basic(data_type: TraceDatasetKey):
    config = get_datagen_config()
    dt = config["dt"] * config["avg_scale"]
    voltages = get_voltage_traces(data_type, config)

    n_to_plot = 100
    dir_path = Path(f"./plots/{data_type}")
    dir_path.mkdir(parents=True, exist_ok=True)
    n_voltages = voltages.shape[0]
    print(f"n_voltages={n_voltages}")
    for i_f in jnp.linspace(0, n_voltages, n_to_plot, endpoint=False):
        i = int(i_f)
        print(f"plotting {i}")
        _plot_trace_basic(
            voltages[i],
            dt,
            dir_path / f"{i:02}.pdf",
        )


def plot_trace_and_reconstruction(
    norm_trace: Array, vae: VAE, dt: int, save_path: Path, take_mean: bool
):
    time_array = jnp.arange(0, norm_trace.shape[0]) * dt
    fig, ax = plt.subplots(figsize=(4, 3))
    if take_mean:
        reconstruction = vae.predict(norm_trace)
    else:
        reconstruction, _, _ = vae(norm_trace, jax.random.PRNGKey(10))

    ax.set_title("Voltage Trace and Reconstruction")
    ax.set_ylabel("Voltage")
    ax.set_xlabel("Time (ms)")
    ax.plot(time_array, norm_trace, label="Original Trace")
    ax.plot(time_array, reconstruction, label="Reconstruction")

    fig.tight_layout()
    fig.legend()
    fig.savefig(save_path)
    plt.close(fig)


def plot_losses(
    channel_type: TraceDatasetKey,
    vae_class: Type[VAE],
    model_date: str,
):
    if model_date is None:
        model_date = get_latest_datestr(channel_type, vae_class)

    model_dir = Path(
        f"./data/models/vae/{channel_type}/{vae_class.__name__}/{model_date}"
    )
    model_config_path = model_dir / "config.yaml"

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    epochs = model_config["epochs"]
    vals_per_epoch = model_config["vals_per_epoch"]

    train_losses = np.load(model_dir / "train_losses.npz")["arr_0"]
    val_losses = np.load(model_dir / "val_losses.npz")["arr_0"]
    nsteps = len(train_losses)
    step_array = np.arange(nsteps)
    train_losses = np.where(train_losses > 1, 1, train_losses)
    total_vals = epochs * vals_per_epoch
    val_steps = np.arange(nsteps, step=np.ceil(nsteps / total_vals).astype(int))
    val_steps = np.concatenate([val_steps, [nsteps - 1]])

    plots_save_path = Path(
        f"./plots/reconstructions/{channel_type}/{vae_class.__name__}/{model_date}"
    )
    plots_save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Losses over Training")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.plot(step_array, train_losses, label="Training Loss")
    ax.plot(val_steps, val_losses, label="Validation Loss")

    ax.set_yscale("log")
    fig.tight_layout()
    fig.legend()
    fig.savefig(plots_save_path / "losses.pdf")
    plt.close(fig)


def plot_reconstructions(
    channel_type: TraceDatasetKey,
    vae_class: Type[VAE],
    trace_index: int,
    model_date: str,
    take_mean: bool,
):
    data_config = get_datagen_config()
    traces = get_voltage_traces(channel_type, data_config)

    if model_date is None:
        model_date = get_latest_datestr(channel_type, vae_class)

    model_dir = Path(
        f"./data/models/vae/{channel_type}/{vae_class.__name__}/{model_date}"
    )
    model_config_path = model_dir / "config.yaml"

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    trace = traces[trace_index]
    normalizer_type: TraceNormalizer = model_config["normalizer_type"]
    dt = data_config["dt"]

    normalizer, _ = get_trace_normal_and_denormal(normalizer_type, channel_type)
    norm_trace = normalizer(trace)

    if take_mean:
        plot_save_path = Path(
            f"./plots/reconstructions/{channel_type}/{vae_class.__name__}/{model_date}/trace_index{trace_index}mean"
        )
    else:
        plot_save_path = Path(
            f"./plots/reconstructions/{channel_type}/{vae_class.__name__}/{model_date}/trace_index{trace_index}"
        )
    plot_save_path.mkdir(parents=True, exist_ok=True)

    vae_models = get_available_vae_models(channel_type, vae_class, model_date)

    for model_filename in vae_models:
        vae = load_vae(
            channel_type, vae_class, len(norm_trace), model_date, epoch=model_filename
        )
        plot_trace_and_reconstruction(
            norm_trace,
            vae,
            dt,
            plot_save_path / f"{model_filename[:-4]}.pdf",
            take_mean,
        )


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "data_type",
        type=str,
        help=f"Type of the cell: Possible values: {get_args(TraceDatasetKey)}",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    _gen_plots(args.data_type)
    # plot_traces_basic(args.data_type)
    # plot_reconstructions(
    #     args.channel_type,
    #     TimeVAEBase,
    #     trace_index=0,
    #     model_date="2026_06_04_18_50_43_568540",
    #     sample=False,
    # )


if __name__ == "__main__":
    main()
