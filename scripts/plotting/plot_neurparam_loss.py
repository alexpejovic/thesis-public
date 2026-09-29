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


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "--channel_type",
        type=str,
        default="Pospischil",
        help="Default: Pospischil",
    )
    parser.add_argument(
        "--sweep_date",
        type=str,
        default="latest",
        help="Default: latest",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    time_str = args.sweep_date
    dir_base = Path(f"./data/comp_losses/{args.channel_type}")

    dir_path = dir_base / time_str

    config_path = dir_path / "config.yaml"
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    assert len(sweep_config["params_dict"]) == 1

    losses = np.load(dir_path / "losses.npz")["arr_0"]
    opt_samples = np.load(dir_path / "opt_samples.npz")["arr_0"]
    b_losses = np.load(dir_path / "b_losses.npz")["arr_0"]
    feat_losses = np.load(dir_path / "feat_losses.npz")["arr_0"]
    mses = np.load(dir_path / "mses.npz")["arr_0"]
    # allen_losses = np.load(dir_path / "allen_losses.npz")["arr_0"]
    params_dict = sweep_config["params_dict"]
    params_tuple = tuple(params_dict.items())

    # i = np.argmax(mses)
    # print(i)
    # print(opt_samples[i - 2 : i + 2])
    # print(mses[i - 2 : i + 2])

    # Normalize all the losses
    mses = mses / np.max(mses)
    b_losses = b_losses / np.max(b_losses)
    feat_losses = feat_losses / np.max(feat_losses)
    # allen_losses = allen_losses / np.max(allen_losses)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Loss w.r.t Neuronal Parameter")
    ax.set_ylabel("Loss")
    ax.set_xlabel(f"Parameter value {params_tuple[0][0]} (Sigmoided)")
    ax.axvline(
        x=params_tuple[0][1],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ideal Value",
    )

    # ax.plot(opt_samples, losses, label="Total Loss")
    # ax.plot(opt_samples, b_losses, label="Boundary Loss (Normalized)")
    # ax.plot(opt_samples, feat_losses, label="Feature (VAE) Loss (Normalized)")
    ax.plot(opt_samples, mses, label="MSE Loss (Normalized)")
    # ax.plot(opt_samples, allen_losses, label="Allen Loss (Normalized)")

    fig.legend()
    fig.tight_layout()
    fig.savefig(dir_path / "losses.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
