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
    get_compressed_time_array,
)

# CMAP = plt.get_cmap("Spectral")


def plot_simple_trace(file_name, i, j, simulated=False):
    channel_type = "Pospischil"

    datagen_cfg = get_datagen_config()
    voltages = get_voltage_traces(channel_type, datagen_cfg)

    time_array = get_compressed_time_array(channel_type, datagen_config=datagen_cfg)

    dir_path = Path("./final_plotting/method_overview/svg")
    dir_path.mkdir(parents=True, exist_ok=True)

    c = colours["simulation"] if simulated else colours["data"]

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, ax = plt.subplots(figsize=(2, 1.5))

        ax.plot(time_array, voltages[i * 10 + j], color=c)

        ax.axis("off")

        fig.tight_layout()
        fig.savefig(dir_path / file_name, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


if __name__ == "__main__":
    plot_simple_trace("simple_trace.svg", 1200, 9)
