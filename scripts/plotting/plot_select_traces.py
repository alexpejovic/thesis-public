import argparse
import os
import sys
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from helper import (
    TraceDatasetKey,
    get_all_data,
)
from plot_simulation import (
    get_compressed_currents,
    get_compressed_time_array,
    plot_sweep,
)

CMAP = plt.get_cmap("Spectral")


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
    data_type = args.data_type

    data = get_all_data(data_type)
    voltages = data["voltages"]
    currents = get_compressed_currents(data_type)
    time_array = get_compressed_time_array(data_type)

    dir_path = Path(f"plots/select_traces/{data_type}")
    dir_path.mkdir(parents=True, exist_ok=True)

    for i in range(500, 550):
        start = i * len(currents)
        end = (i + 1) * len(currents)
        voltage_traces = voltages[start:end]
        save_path = dir_path / f"{start:03}-{(end - 1):03}.pdf"
        plot_sweep(voltage_traces, currents, time_array, save_path)


if __name__ == "__main__":
    main()
