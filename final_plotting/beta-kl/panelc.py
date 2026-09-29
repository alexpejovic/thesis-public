# Correlations of each loss function w MSE
import os
import sys
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from shared import ALL_PARAMS, ENCODER_COLOURS, ENCODER_DATES, N_ENCODERS

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent2_dir)

from helper import get_feature_losses

LossType = Literal["jaxley", "mse"]

STEP = 20
LO_PC = 0.0
HI_PC = 99.0


def plot_beta_mse_corr(channel_type, np1, np2, panel_letter):
    target_freeze_params = ALL_PARAMS.copy()
    target_freeze_params.remove(np1)
    target_freeze_params.remove(np2)

    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, axs = plt.subplots(
            1,
            N_ENCODERS,
            figsize=(0.95 * N_ENCODERS, 1.0),
            sharey=True,
            layout="constrained",
        )
        _, _, _, mses, _, _ = get_feature_losses(
            channel_type,
            ENCODER_DATES[0],
            target_freeze_params,
            named_encoder=True,
            feat_loss="kl",
        )

        mses = mses / np.max(mses)
        lo, hi = np.percentile(mses, [LO_PC, HI_PC])
        mses = np.clip((mses - lo / (hi - lo)), 0.0, 1.0)
        mses = mses[::STEP]

        for i in range(N_ENCODERS):
            feat_loss, _, _, _, _, _ = get_feature_losses(
                channel_type,
                ENCODER_DATES[i],
                target_freeze_params,
                named_encoder=True,
                feat_loss="kl",
            )
            lo, hi = np.percentile(feat_loss, [LO_PC, HI_PC])
            feat_loss = np.clip((feat_loss - lo) / (hi - lo), 0.0, 1.0)
            feat_loss = feat_loss[::STEP]
            axs[i].scatter(mses, feat_loss, s=0.2, alpha=0.3, c=ENCODER_COLOURS[i])
            axs[i].axline((0, 0), (1, 1), color="k", linestyle="dotted", alpha=0.3)

        axs[0].set_ylabel("Enc loss (norm)")
        fig.supxlabel(
            "MSE (normalized)",
            fontproperties=axs[0].title.get_fontproperties(),
            x=0.55,
            y=-0.06,
        )
        # axs[1].set_xlabel("MSE (normalized)")
        save_path = Path(f"./final_plotting/beta-kl/panels/panel{panel_letter}.svg")
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    plot_beta_mse_corr("Pospischil", "Na_gNa", "K_gK", "c")
