import os
import sys
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from panela import LATEX_NAMES

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import get_feature_losses
from matplotlib.colors import LinearSegmentedColormap

ENCODER_DATES = ["2026_08_12_15_20_52_343885"]

ALL_PARAMS = [
    "Leak_gLeak",
    "Leak_eLeak",
    "Na_gNa",
    "K_gK",
    "Km_gKm",
    "CaL_gCaL",
    "Km_taumax",
    "vt",
]

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
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(2.4, 1.2),
            gridspec_kw={"width_ratios": [1, 1]},
            sharey=True,
            layout="constrained",
        )

        _, opt_samples, pt, mses, _, jaxley_losses = get_feature_losses(
            channel_type, ENCODER_DATES[0], target_freeze_params
        )
        d = len(opt_samples)
        mses = mses.reshape((d, d))

        # Normalize both losses
        mses = mses / np.max(mses)
        jaxley_losses = jaxley_losses / np.max(jaxley_losses)
        mses = mses.reshape((d, d))
        jaxley_losses = jaxley_losses.reshape((d, d))

        o_lower = np.min(opt_samples)
        o_upper = np.max(opt_samples)

        cmap_mse = LinearSegmentedColormap.from_list(
            "loss",
            ["white", colours["mse"]],
        )
        cmap_jaxley = LinearSegmentedColormap.from_list(
            "loss",
            ["white", colours["jaxley_loss"]],
        )

        im_0 = axs[0].imshow(
            mses,
            origin="lower",
            extent=(o_lower, o_upper, o_lower, o_upper),
            aspect="auto",
            cmap=cmap_mse,
        )

        im_1 = axs[1].imshow(
            jaxley_losses,
            origin="lower",
            extent=(o_lower, o_upper, o_lower, o_upper),
            aspect="auto",
            cmap=cmap_jaxley,
        )
        axs[0].set(
            ylabel=rf"{LATEX_NAMES[paramy]} (sigmoid)",
        )

        cax_0 = fig.add_axes((1.00, 0.26, 0.02, 0.71))
        cax_1 = fig.add_axes((1.03, 0.26, 0.02, 0.71))

        cbar_0 = fig.colorbar(im_0, cax=cax_0)
        cbar_1 = fig.colorbar(im_1, cax=cax_1)

        cbar_0.set_ticks([])
        cbar_1.set_ticks([0.0, 1.0])
        cbar_1.set_ticklabels(["low", "high"])
        cbar_1.set_label("Loss", rotation=270)

        xlim = axs[0].get_xlim()
        ylim = axs[0].get_ylim()

        axs[0].scatter(pt[0][1], pt[1][1], c=colours["ideal"], s=1)
        axs[1].scatter(pt[0][1], pt[1][1], c=colours["ideal"], s=1)

        axs[1].set_xlim(xlim)
        axs[1].set_ylim(ylim)

        fig.supxlabel(
            rf"{LATEX_NAMES[paramx]} (sigmoid)",
            fontproperties=axs[0].title.get_fontproperties(),
            x=0.55,
        )
        save_path = Path(f"./final_plotting/baselines/panels/panel{panel_letter}.svg")
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    plot_2d_loss("Pospischil", "Na_gNa", "Km_gKm", "c")
