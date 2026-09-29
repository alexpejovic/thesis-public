import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_feat_sweep_per_encoder import ALL_PARAMS, get_feature_losses

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from vae.conv_vae import TimeVAEBase
from vae.vae_helper import get_vae_config


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "--neur_param1",
        type=str,
        default="Na_gNa",
        help="Default: Na_gNa",
    )
    parser.add_argument(
        "--neur_param2",
        type=str,
        default="K_gK",
        help="Default: K_gK",
    )
    parser.add_argument(
        "--encoder_date",
        type=str,
        default="2026_07_22_12_23_35_443907",
        help="Default: 2026_07_22_12_23_35_443907",
    )
    parser.add_argument(
        "--plot_allen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default: --no-plot_allen",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    channel_type = "Pospischil"
    neur_param1 = args.neur_param1
    neur_param2 = args.neur_param2
    encoder_date = args.encoder_date
    plot_allen = args.plot_allen

    param1_i = ALL_PARAMS.index(neur_param1)
    param2_i = ALL_PARAMS.index(neur_param2)

    if param1_i < param2_i:
        paramx = neur_param1
        paramy = neur_param2
    else:
        paramx = neur_param2
        paramy = neur_param1

    target_freeze_params = ALL_PARAMS.copy()
    target_freeze_params.remove(paramx)
    target_freeze_params.remove(paramy)

    feat_losses, opt_samples, _, mses, allen_losses, _ = get_feature_losses(
        channel_type, encoder_date, target_freeze_params
    )

    vae_conf = get_vae_config("Pospischil", TimeVAEBase, encoder_date)
    try:
        beta = vae_conf["beta"]
    except KeyError:
        beta = vae_conf["beta_norm"]
    latent_dim = vae_conf["latent_dim"]
    activation = vae_conf["activation"]
    normalizer = vae_conf["normalizer_type"]
    try:
        masking = vae_conf["masking"]
    except KeyError:
        masking = False

    d = len(opt_samples)
    X = np.tile(opt_samples, d).reshape((d, d))
    Y = opt_samples.repeat(d).reshape((d, d))
    feat_losses = feat_losses.reshape((d, d))
    mses = mses.reshape((d, d))

    # Normalize both losses
    feat_losses = feat_losses / np.max(feat_losses)
    mses = mses / np.max(mses)

    z_max = 1.0
    o_lower = np.min(opt_samples)
    o_upper = np.max(opt_samples)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot_surface(
        X,
        Y,
        feat_losses,
        edgecolor="royalblue",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
        label="Feature Loss",
    )

    ax.plot_surface(
        X,
        Y,
        mses,
        edgecolor="indianred",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
        label="MSE",
    )

    if plot_allen:
        allen_losses = allen_losses.reshape((d, d))
        allen_losses = allen_losses / np.max(allen_losses)
        ax.plot_surface(
            X,
            Y,
            allen_losses,
            edgecolor="darkolivegreen",
            lw=0.5,
            rstride=8,
            cstride=8,
            alpha=0.3,
            label="Allen",
        )

    # ax.contour(X, Y, Z, zdir="z", offset=0, cmap="coolwarm")
    # ax.contour(X, Y, Z, zdir="x", offset=o_lower, cmap="coolwarm")
    # ax.contour(X, Y, Z, zdir="y", offset=o_upper, cmap="coolwarm")

    ax.set(
        xlim=(o_lower, o_upper),
        ylim=(o_lower, o_upper),
        zlim=(0, z_max),
        xlabel=f"{paramx}",
        ylabel=f"{paramy}",
        zlabel="Loss (Normalized)",
    )
    if masking:
        fig.suptitle(
            f"Beta:{beta}, Latent Dim:{latent_dim}, Activation: {activation}, Norm: {normalizer}, With Masking"
        )
    else:
        fig.suptitle(
            f"Beta:{beta}, Latent Dim:{latent_dim}, Activation: {activation}, Norm: {normalizer}"
        )

    if plot_allen:
        save_path = Path(
            f"plots/encoder_mse_3d/{encoder_date}/with_allen/{paramx}-{paramy}.pdf"
        )
    else:
        save_path = Path(f"plots/encoder_mse_3d/{encoder_date}/{paramx}-{paramy}.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    main()

    print("Complete!")
