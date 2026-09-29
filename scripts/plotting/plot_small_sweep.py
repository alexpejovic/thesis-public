import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from helper import compress_traces


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "--channel_type",
        type=str,
        default="Pospischil",
        help="Default: Pospischil",
    )
    parser.add_argument(
        "--sweep_date",
        type=str,
        default="latest",
        help="Default: latest",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    time_str = args.sweep_date
    dir_base = Path(f"./data/comp_small_sweeps/{args.channel_type}")

    dir_path = dir_base / time_str

    voltage_array = np.load(dir_path / "voltages.npz")["arr_0"]
    opt_samples = np.load(dir_path / "opt_samples.npz")["arr_0"]
    ideal_voltages = np.load(dir_path / "ideal_voltages.npz")["arr_0"]
    time_array = np.arange(0, voltage_array.shape[2]) * 0.025

    save_path = dir_path / "compare.pdf"

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, opt_sample in enumerate(opt_samples):
        voltages = voltage_array[i]
        save_path = dir_path / f"opt_sample{opt_sample}.pdf"

        fig, ax = plt.subplots(figsize=(8, 6))
        voltages_compressed = compress_traces(voltages, 10.0, 0.025, 4)
        voltages_compressed = np.nan_to_num(voltages_compressed)
        mse = np.mean((ideal_voltages - voltages_compressed) ** 2)
        print(opt_sample)
        print(mse)

        ax.set_title(f"Voltage Trace for opt:{opt_sample:.4f} mse:{mse:.4f}")
        ax.set_ylabel("Voltage")
        ax.set_xlabel("Time (ms)")

        for voltage_trace in voltages:
            ax.plot(time_array, voltage_trace)
        # ax.plot(time_array, voltages[0])

        fig.tight_layout()
        fig.legend()
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
