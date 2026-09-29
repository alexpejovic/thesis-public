import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from colours import colours
from helper import get_datagen_config, get_feature_losses
from train_comp import _get_transform_list

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

SHORT = {
    "Leak_gLeak": "gLeak",
    "Leak_eLeak": "eLeak",
    "Na_gNa": "gNa",
    "K_gK": "gK",
    "Km_gKm": "gKm",
    "CaL_gCaL": "gCaL",
    "Km_taumax": "taumax",
    "vt": "vt",
}

LATEX_NAMES = {
    "Leak_gLeak": r"$g_{Leak}$",
    "Leak_eLeak": r"$e_{Leak}$",
    "Na_gNa": r"$g_{Na}$",
    "K_gK": r"$g_{K}$",
    "Km_gKm": r"$g_{Km}$",
    "CaL_gCaL": r"$g_{CaL}$",
    "Km_taumax": r"$\tau_{max}$",
    "vt": r"$V_T$",
}


def plot_mse_and_jaxley(channel_type, neur_param, panel_letter, with_legend=True):
    target_freeze_params = ALL_PARAMS.copy()
    target_freeze_params.remove(neur_param)

    datagen_cfg = get_datagen_config()
    bounds_dict = {
        neur_param: datagen_cfg["param_bounds"][channel_type][SHORT[neur_param]]
    }
    transform_list = _get_transform_list(bounds_dict)

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        # ax.set_ylabel("Loss")
        ax.set_yticks([])
        ax.set_xlabel(rf"{LATEX_NAMES[neur_param]} (sigmoid)")
        ax.set_title("Loss Shape")

        _, _, pt, _, _, _ = get_feature_losses(
            channel_type, ENCODER_DATES[0], target_freeze_params
        )
        ideal_value = pt[0][1]
        ax.axvline(
            x=ideal_value,
            color=colours["ideal"],
            linestyle="--",
            linewidth=1,
            label="Ideal Value",
        )

        for i, encoder_date in enumerate(ENCODER_DATES):
            _, opt_samples, _, mses, _, jaxley_losses = get_feature_losses(
                channel_type, encoder_date, target_freeze_params
            )
            # opt_transformed = _forward_transform_params([opt_samples], transform_list)[0]

            if i == 0:
                # mses = (mses / np.max(mses)) - (i / 2)
                mses = mses / np.max(mses)
                jaxley_losses = jaxley_losses / np.max(jaxley_losses)
                ax.plot(opt_samples, mses, label="MSE", color=colours["mse"])
                ax.plot(
                    opt_samples,
                    jaxley_losses,
                    label="Jaxley",
                    color=colours["jaxley_loss"],
                )

        save_path = Path(f"./final_plotting/baselines/panels/panel{panel_letter}.svg")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if with_legend:
            leg = fig.legend()

            plt.draw()  # Draw the figure so you can find the positon of the legend.

            # Get the bounding box of the original legend
            bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())

            # # Change to location of the legend.
            xOffset = 0.65
            bb.x0 += xOffset
            bb.x1 += xOffset
            yOffset = 0.29
            bb.y0 += yOffset
            bb.y1 += yOffset
            leg.set_bbox_to_anchor(bb, transform=ax.transAxes)
            ax.set_ylabel("Loss")

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    plot_mse_and_jaxley("Pospischil", "Na_gNa", "a")
