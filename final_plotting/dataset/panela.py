import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import (
    get_datagen_config,
    get_voltage_traces,
)
from plot_simulation import (
    get_compressed_currents,
    get_compressed_time_array,
)

# CMAP = plt.get_cmap("Spectral")


def plot_sweep(channel_type, file_name):
    datagen_cfg = get_datagen_config()
    voltages = get_voltage_traces(channel_type, datagen_cfg)

    time_array = get_compressed_time_array(channel_type, datagen_config=datagen_cfg)
    currents = get_compressed_currents(channel_type, datagen_cfg)

    dir_path = Path("./final_plotting/dataset/panels")
    dir_path.mkdir(parents=True, exist_ok=True)

    # colors = CMAP(jnp.linspace(0, 1, len(currents)))

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(3, 2.25), gridspec_kw={"height_ratios": [3, 1]}
        )

        j = 9

        ax1.plot(time_array, voltages[j], color=colours["simulation"])

        ax1.sharex(ax2)
        ax1.tick_params(labelbottom=False)
        ax1.set_ylabel("Voltage (mV)")
        # ax1.set_title("Voltage and Current Traces")
        ax2.plot(time_array, currents[j] * 1000.0, color=colours["simulation"])
        ax2.set_ylabel("Current (pA)")
        ax2.set_xlabel("Time (ms)")

        fig.tight_layout()
        fig.savefig(dir_path / file_name)
        plt.close(fig)


if __name__ == "__main__":
    plot_sweep("Pospischil", "panela.svg")
