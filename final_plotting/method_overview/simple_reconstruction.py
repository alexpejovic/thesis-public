import os
import sys
from pathlib import Path

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import (
    TraceNormalizer,
    get_datagen_config,
    get_trace_normal_and_denormal,
    get_voltage_traces,
)
from plot_simulation import get_compressed_time_array
from vae.conv_vae import TimeVAEBase
from vae.vae_helper import (
    load_named_vae,
)


def plot_named_reconstruction_simple(
    model_name: str,
    take_mean: bool,
):
    channel_type = "Pospischil"
    file_name = "simple_reconstruction.svg"
    vae_class = TimeVAEBase
    i = 1200
    j = 9
    trace_index = 10 * i + j

    data_config = get_datagen_config()
    traces = get_voltage_traces(channel_type, data_config)

    model_dir = Path(f"./data/models/vae/named/{model_name}")
    model_config_path = model_dir / "config.yaml"

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    trace = traces[trace_index]
    normalizer_type: TraceNormalizer = model_config["normalizer_type"]
    time_array = get_compressed_time_array(channel_type)

    normalizer, _ = get_trace_normal_and_denormal(normalizer_type, channel_type)
    norm_trace = normalizer(trace)

    plot_save_path = Path("./final_plotting/method_overview/svg")
    plot_save_path.mkdir(parents=True, exist_ok=True)

    vae = load_named_vae(vae_class, len(norm_trace), model_name, epoch=None)
    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, ax = plt.subplots(figsize=(2.0, 1.5))
        if take_mean:
            reconstruction = vae.predict(norm_trace)
        else:
            reconstruction, _, _ = vae(norm_trace, jax.random.PRNGKey(10))

        ax.set_ylabel("Normalized Voltage")
        ax.set_xlabel("Time (ms)")
        ax.plot(time_array, norm_trace, label="Original Trace", color=colours["data"])
        ax.plot(
            time_array,
            reconstruction,
            label="Reconstruction",
            color=colours["reconstruction"],
        )

        # Remove axis labels
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # ax.legend(loc='upper right')

        fig.tight_layout()
        fig.savefig(plot_save_path / file_name, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


if __name__ == "__main__":
    plot_named_reconstruction_simple("gelu-beta0.2", True)
