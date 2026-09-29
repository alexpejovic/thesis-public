# Bar Plot with the number of local minima
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shared import ALL_PARAMS, ENCODER_COLOURS, ENCODER_DATES, ENCODER_NAMES, N_ENCODERS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import get_feature_losses


def n_local_min(a):
    return np.sum((a[1:-1] < a[:-2]) & (a[1:-1] < a[2:]))


def plot_local_min_bar(channel_type, neur_param):
    target_freeze_params = ALL_PARAMS.copy()
    target_freeze_params.remove(neur_param)

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, ax = plt.subplots(figsize=(0.95, 0.95))

        _, _, _, mses, _, _ = get_feature_losses(
            channel_type,
            ENCODER_DATES[0],
            target_freeze_params,
            named_encoder=True,
        )
        local_mins = [n_local_min(mses)]
        labels = ["MSE"] + list(ENCODER_NAMES.values())
        cols = [colours["mse"]] + ENCODER_COLOURS

        for i in range(N_ENCODERS):
            feat_loss, _, _, _, _, _ = get_feature_losses(
                channel_type,
                ENCODER_DATES[i],
                target_freeze_params,
                named_encoder=True,
            )
            local_mins.append(n_local_min(feat_loss))

        ax.bar(labels, local_mins, color=cols)
        # ax.xlabel("Category")
        # ax.ylabel("Value")
        # ax.title("Simple Bar Plot")
        save_path = Path("./final_plotting/masking/panels/paneld.svg")
        ax.set_xticks([])
        ax.set_ylabel("# Local Minima")
        fig.savefig(save_path)
        plt.close(fig)

    print(local_mins)


if __name__ == "__main__":
    plot_local_min_bar("Pospischil", "Na_gNa")
