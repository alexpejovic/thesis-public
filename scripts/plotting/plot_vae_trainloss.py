import argparse
import os
import sys
from typing import get_args

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)


from helper import ChannelType
from plot_simulation import plot_losses
from vae.vae_helper import MODEL_STRINGS


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "--channel_type",
        type=str,
        default=(default := "Pospischil"),
        help=f"Type of the cell: Possible values: {get_args(ChannelType)}. Default: {default}.",
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default=(default := "TimeVAEBase"),
        help=f"Type of VAE: Possible values: {MODEL_STRINGS.keys()}. Default: {default}.",
    )
    parser.add_argument(
        "--model_date",
        type=str,
        default=(default := "latest"),
        help=f"Timestamp of the model. Default {default}.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    plot_losses(
        args.channel_type,
        MODEL_STRINGS[args.vae_type],
        model_date=args.model_date,
    )
    print("Complete!")


if __name__ == "__main__":
    main()
