import argparse
import os
import sys
from typing import get_args

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)


from helper import ChannelType
from plot_simulation import plot_reconstructions
from vae.vae_helper import MODEL_STRINGS


def parse_args() -> dict:
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
        "--trace_index",
        type=int,
        default=(default := 0),
        help=f"Index of the trace you want to reconstruct. Default {default}.",
    )
    parser.add_argument(
        "--model_date",
        type=str,
        default=(default := "latest"),
        help=f"Timestamp of the model. Default {default}.",
    )
    parser.add_argument(
        "-tm",
        "--take_mean",
        action="store_true",
        help="Take the mean of the latent distribution (as opposed to sampling from it)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    plot_reconstructions(
        args.channel_type,
        MODEL_STRINGS[args.vae_type],
        trace_index=args.trace_index,
        model_date=args.model_date,
        take_mean=args.take_mean,
    )


if __name__ == "__main__":
    main()
