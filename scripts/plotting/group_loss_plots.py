import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal, get_args

from pypdf import PdfWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

HyperParam = Literal["latent_dim", "activation", "beta"]


def _group_loss_plots(
    hyper_param: HyperParam,
    param_value: str,
):
    config_files_glob = glob.glob(
        "data/models/vae/Pospischil/TimeVAEBase/2026_08*/config.yaml"
    )

    grep_sb = subprocess.run(
        [
            "grep",
            "-l",
            f"{hyper_param}: {param_value}$",
            *config_files_glob,
        ],
        capture_output=True,
        text=True,
    )
    filename_list = grep_sb.stdout.split("\n")
    filename_list.remove("")
    filename_list = list(filter(lambda s: "latest" not in s, filename_list))

    datestr_list = [re.search(r"(2026_08.*)/", f).groups()[0] for f in filename_list]
    loss_paths = [
        f"./plots/reconstructions/Pospischil/TimeVAEBase/{s}/losses.pdf"
        for s in datestr_list
    ]

    merger = PdfWriter()
    # Append multiple PDF files
    for pdf in loss_paths:
        merger.append(pdf)

    combined_dir = Path(f"./plots/vae_training_grouped/{hyper_param}")
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Write the combined result
    merger.write(combined_dir / f"{param_value}.pdf")
    merger.close()


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process cell type input.")
    parser.add_argument(
        "--hyper_param",
        type=str,
        default=(default := "latent_dim"),
        help=f"Type of the cell: Possible values: {get_args(HyperParam)}. Default: {default}.",
    )
    parser.add_argument(
        "--param_value",
        type=str,
        default=(default := "8"),
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    _group_loss_plots(
        args.hyper_param,
        args.param_value,
    )
    print("Complete!")


if __name__ == "__main__":
    main()
