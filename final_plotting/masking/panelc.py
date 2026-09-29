# Correlations of each loss function w MSE
import os
import sys
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shared import (
    ALL_PARAMS,
    ENCODER_COLOURS,
    ENCODER_DATES,
    MASKED,
    N_ENCODERS,
    NON_MASKED,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from helper import get_feature_losses
from colours import colours

LossType = Literal["jaxley", "mse"]

STEP = 20
LO_PC = 0.0
HI_PC = 99.0


def plot_masking(channel_type, np1, np2, panel_letter):
    target_freeze_params = ALL_PARAMS.copy()
    target_freeze_params.remove(np1)
    target_freeze_params.remove(np2)

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(0.95 * N_ENCODERS, 0.95),
            sharey=True,
            layout="constrained",
        )

        _, _, _, mses, _, _ = get_feature_losses(
            channel_type,
            ENCODER_DATES[0],
            target_freeze_params,
            named_encoder=True,
        )

        mses = mses / np.max(mses)
        lo, hi = np.percentile(mses, [LO_PC, HI_PC])
        mses = np.clip((mses - lo / (hi - lo)), 0.0, 1.0)
        mses = mses[::STEP]

        masked_loss, _, _, _, _, _ = get_feature_losses(
            channel_type,
            MASKED,
            target_freeze_params,
            named_encoder=True,
        )
        lo, hi = np.percentile(masked_loss, [LO_PC, HI_PC])
        masked_loss = np.clip((masked_loss - lo) / (hi - lo), 0.0, 1.0)
        masked_loss = masked_loss[::STEP]

        unmasked_loss, _, _, _, _, _ = get_feature_losses(
            channel_type,
            NON_MASKED,
            target_freeze_params,
            named_encoder=True,
        )
        lo, hi = np.percentile(unmasked_loss, [LO_PC, HI_PC])
        unmasked_loss = np.clip((unmasked_loss - lo) / (hi - lo), 0.0, 1.0)
        unmasked_loss = unmasked_loss[::STEP]

        axs[0].scatter(masked_loss, mses, s=0.2, alpha=0.3, c=colours["mse"])
        axs[0].axline((0, 0), (1, 1), color="k", linestyle="dotted", alpha=0.3)
        axs[1].scatter(
            masked_loss, unmasked_loss, s=0.2, alpha=0.3, c=ENCODER_COLOURS[1]
        )
        axs[1].axline((0, 0), (1, 1), color="k", linestyle="dotted", alpha=0.3)

        axs[0].set_ylabel("Other Loss")
        fig.supxlabel(
            "Masked Enc Loss (normalized)",
            fontproperties=axs[0].title.get_fontproperties(),
            x=0.60,
            y=-0.06,
        )
        # axs[1].set_xlabel("MSE (normalized)")
        save_path = Path(f"./final_plotting/masking/panels/panel{panel_letter}.svg")
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    plot_masking("Pospischil", "Na_gNa", "K_gK", "c")
