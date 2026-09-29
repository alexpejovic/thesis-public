import os
import sys
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shared import ALL_PARAMS, ENCODER_COLOURS, ENCODER_DATES, N_ENCODERS, SHORT

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import get_feature_losses
from matplotlib.colors import LinearSegmentedColormap

LossType = Literal["jaxley", "mse"]


def plot_2d_loss(channel_type, np1, np2, panel_letter):
    target_freeze_params = ALL_PARAMS.copy()
    target_freeze_params.remove(np1)
    target_freeze_params.remove(np2)

    param1_i = ALL_PARAMS.index(np1)
    param2_i = ALL_PARAMS.index(np2)

    if param1_i < param2_i:
        paramx = np1
        paramy = np2
    else:
        paramx = np2
        paramy = np1

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        width, height = 1.6, 1.5
        # y = np.sqrt(height) / 100 * 0.2
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(width, height),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        _, opt_samples, pt, _, _, _ = get_feature_losses(
            channel_type,
            ENCODER_DATES[0],
            target_freeze_params,
            named_encoder=True,
        )

        d = len(opt_samples)

        o_lower = np.min(opt_samples)
        o_upper = np.max(opt_samples)

        encoder_cmaps = [
            LinearSegmentedColormap.from_list("loss", ["white", ENCODER_COLOURS[i]])
            for i in range(N_ENCODERS)
        ]

        ims = []

        for i in range(N_ENCODERS):
            feat_loss, _, _, _, _, _ = get_feature_losses(
                channel_type,
                ENCODER_DATES[i],
                target_freeze_params,
                named_encoder=True,
            )
            feat_loss = feat_loss.reshape((d, d))
            feat_loss = feat_loss / np.max(feat_loss)
            im = axs[i // 2][i % 2].imshow(
                feat_loss,
                origin="lower",
                extent=(o_lower, o_upper, o_lower, o_upper),
                aspect="auto",
                cmap=encoder_cmaps[i],
            )
            ims.append(im)

        # axs[0][0].set(
        #     ylabel=f"{paramy} (sigmoid)",
        # )

        for i in range(N_ENCODERS):
            cax = fig.add_axes((1.00 + 0.042 * i, 0.14, 0.03, 0.825))
            cbar = fig.colorbar(ims[i], cax=cax)
            if i == N_ENCODERS - 1:
                cbar.set_ticks([0.0, 1.0])
                cbar.set_ticklabels(["low", "high"])
                cbar.set_label("Loss", rotation=270)
            else:
                cbar.set_ticks([])

        xlim = axs[0][0].get_xlim()
        ylim = axs[0][0].get_ylim()

        for i in range(N_ENCODERS):
            axs[i // 2][i % 2].scatter(pt[0][1], pt[1][1], c=colours["ideal"], s=1)

        for i in range(1, N_ENCODERS):
            axs[i // 2][i % 2].set_xlim(xlim)
            axs[i // 2][i % 2].set_ylim(ylim)

        fig.supxlabel(
            rf"{SHORT[paramx]} (sigmoid)",
            fontproperties=axs[0][0].title.get_fontproperties(),
            x=0.60,
            y=-0.06,
        )
        # axs[1][0].set_xlabel(f"{paramx} (sigmoid)")
        # axs[1][0].xaxis.set_label_position("right")
        fig.supylabel(
            rf"{SHORT[paramy]} (sigmoid)",
            fontproperties=axs[0][0].title.get_fontproperties(),
            x=-0.06,
            y=0.60,
        )
        # plt.subplots_adjust(bottom=0.15)
        save_path = Path(f"./final_plotting/beta/panels/panel{panel_letter}.svg")
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    plot_2d_loss("Pospischil", "Na_gNa", "K_gK", "b")
