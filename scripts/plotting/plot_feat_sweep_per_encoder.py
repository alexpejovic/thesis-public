import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from helper import get_feature_losses
from vae.conv_vae import TimeVAEBase

# ENCODER_DATES = [
#     # "2026_07_17_15_15_02_272928",
#     # "2026_07_17_15_16_31_982382",
#     # "2026_07_17_15_16_42_852332",
#     "2026_07_17_15_16_42_852314",
#     "2026_07_17_15_16_42_852296",
#     "2026_07_17_15_17_01_801707",
#     "2026_07_17_15_19_33_752590",
#     # "2026_07_17_15_20_33_705272",
# ]

ENCODER_DATES = [
    "2026_07_17_15_15_02_272928",
    "2026_07_17_15_16_31_982382",
    "2026_07_17_15_16_42_852332",
    "2026_07_17_15_16_42_852314",
    "2026_07_22_12_23_38_145434",
    "2026_07_22_12_23_35_443907",
    # "2026_07_22_12_23_37_369014",
    # "2026_07_22_12_23_36_486258",
    # "2026_07_22_12_27_31_796950",
    "2026_07_22_12_29_47_076872",
    # "2026_07_22_12_38_07_493078",
    # "2026_07_22_12_38_47_147985",
    # "2026_07_22_12_42_42_277746",
    "2026_07_22_12_43_17_787259",
]

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


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "--neur_param",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    channel_type = "Pospischil"
    neur_param = args.neur_param

    target_freeze_params = ALL_PARAMS.copy()
    target_freeze_params.remove(neur_param)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Loss Shape w.r.t Encoder Beta")
    # ax.set_ylabel("Loss")
    ax.set_yticks([])
    ax.set_xlabel(f"Parameter value {neur_param} (Sigmoided)")

    _, _, pt, _, _, _ = get_feature_losses(
        channel_type, ENCODER_DATES[0], target_freeze_params
    )
    ideal_value = pt[0][1]
    ax.axvline(
        x=ideal_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ideal Value",
    )
    CMAP = plt.get_cmap("inferno")
    colors = CMAP(np.linspace(0, 0.9, len(ENCODER_DATES)))

    for i, encoder_date in enumerate(ENCODER_DATES):
        encoder_config = Path(
            f"./data/models/vae/{channel_type}/{TimeVAEBase.__name__}/{encoder_date}/config.yaml"
        )
        with open(encoder_config, "r") as f:
            try:
                d = yaml.safe_load(f)
                beta = d["beta"]
            except KeyError:
                beta_norm = d["beta_norm"]
                # WATCH OUT FOR THIS HARDCODING
                beta = beta_norm * 1300 / 64
        feat_loss, opt_samples, _, mses, _, _ = get_feature_losses(
            channel_type, encoder_date, target_freeze_params
        )
        feat_loss = (feat_loss / np.max(feat_loss)) - ((i + 1) / 2)

        if i == 0:
            mses = (mses / np.max(mses)) - (i / 2)
            ax.plot(opt_samples, mses, label="MSE", color="g")

        ax.plot(opt_samples, feat_loss, label=f"beta={beta:.3f}", color=colors[i])

    save_path = Path(f"plots/beta_to_loss/encoder_shapes/{neur_param}.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
