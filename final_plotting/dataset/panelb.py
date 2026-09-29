import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)


from helper import (
    get_datagen_config,
    get_voltage_traces,
)
from plot_simulation import (
    get_compressed_currents,
    get_compressed_time_array,
)
from panela import plot_sweep



if __name__ == "__main__":
    plot_sweep("Pospischil-bad", "panelb.svg")
