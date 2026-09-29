# Bar Plot with the number of local minima
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from shared import ALL_PARAMS, ENCODER_COLOURS, ENCODER_DATES, ENCODER_NAMES, N_ENCODERS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import get_opt_sweep_score

# Plot the optimization scores for each mask


def plot_opt_bar(channel_type, neur_params):
    target_freeze_params = ALL_PARAMS.copy()
    for neur_param in neur_params:
        target_freeze_params.remove(neur_param)

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, ax = plt.subplots(figsize=(0.95, 0.95))

        opt_metric_mse = get_opt_sweep_score(
            channel_type,
            None,
            target_freeze_params,
            target_feat_loss="mse",
            target_optimizer="jaxley_polyak",
        )
        labels = ["MSE"] + list(ENCODER_NAMES.values())
        cols = [colours["mse"]] + ENCODER_COLOURS
        opt_metrics = [opt_metric_mse]

        for i in range(N_ENCODERS):
            opt_metric = get_opt_sweep_score(
                channel_type,
                ENCODER_DATES[i],
                target_freeze_params,
                named_encoder=True,
                # target_optimizer="jaxley_polyak",
            )
            opt_metrics.append(opt_metric)

        ax.bar(labels, opt_metrics, color=cols)
        # ax.xlabel("Category")
        # ax.ylabel("Value")
        # ax.title("Simple Bar Plot")
        save_path = Path("./final_plotting/masking/panels/panele.svg")
        ax.set_xticks([])
        ax.set_ylabel("Opt Score")
        fig.savefig(save_path)
        plt.close(fig)

    print(opt_metrics)


if __name__ == "__main__":
    plot_opt_bar("Pospischil", ["Na_gNa", "K_gK"])
