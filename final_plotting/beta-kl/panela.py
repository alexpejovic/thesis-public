import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shared import ALL_PARAMS, ENCODER_COLOURS, ENCODER_DATES, ENCODER_NAMES

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import get_feature_losses


def plot_1d_mse_compare(channel_type, neur_params, panel_letter):
    n_params = len(neur_params)

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, axs = plt.subplots(1, 2, figsize=(1.15 * n_params, 1.85), sharey=True)
        for i, neur_param in enumerate(neur_params):
            target_freeze_params = ALL_PARAMS.copy()
            target_freeze_params.remove(neur_param)

            axs[i].set_yticks([])
            axs[i].set_xlabel(f"{neur_param} (sigmoid)")

            # print(ENCODER_DATES[0])
            _, opt_samples, pt, mses, _, _ = get_feature_losses(
                channel_type,
                ENCODER_DATES[0],
                target_freeze_params,
                named_encoder=True,
                feat_loss="kl",
            )
            ideal_value = pt[0][1]
            axs[i].axvline(
                x=ideal_value,
                color=colours["ideal"],
                linestyle="--",
                linewidth=1,
                label="Ideal Value" if i == 0 else "",
            )
            mses = mses / np.max(mses)
            axs[i].plot(
                opt_samples, mses, label="MSE" if i == 0 else "", color=colours["mse"]
            )

            for j, encoder_date in enumerate(ENCODER_DATES):
                print(encoder_date)
                feat_loss, _, _, _, _, _ = get_feature_losses(
                    channel_type,
                    encoder_date,
                    target_freeze_params,
                    named_encoder=True,
                    feat_loss="kl",
                )

                feat_loss = (feat_loss / np.max(feat_loss)) - ((j + 1) * 0.25)
                axs[i].plot(
                    opt_samples,
                    feat_loss,
                    label=ENCODER_NAMES[encoder_date] if i == 0 else "",
                    c=ENCODER_COLOURS[j],
                )

        axs[0].set_ylabel("Losses (shifted)")
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)

        # leg = fig.legend()

        # plt.draw()  # Draw the figure so you can find the positon of the legend.
        #
        # # Get the bounding box of the original legend
        # bb = leg.get_bbox_to_anchor().transformed(axs[i].transAxes.inverted())
        #
        # # # Change to location of the legend.
        # xOffset = 0.0
        # bb.x0 += xOffset
        # bb.x1 += xOffset
        # yOffset = 0.00
        # bb.y0 += yOffset
        # bb.y1 += yOffset
        # leg.set_bbox_to_anchor(bb, transform=axs[i].transAxes)

        save_path = Path(f"./final_plotting/beta-kl/panels/panel{panel_letter}.svg")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # fig.tight_layout(rect=(-0.5, 1, 1, 1))
        fig.tight_layout(rect=(0, 0, 1, 0.75))
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    plot_1d_mse_compare("Pospischil", ["Na_gNa", "K_gK"], "a")
