import os
import sys

import matplotlib as mpl

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from colours import colours

MASKED = "mask-beta0.5"
NON_MASKED = "gelu-beta0.5"

ENCODER_DATES = [MASKED, NON_MASKED]

ENCODER_NAMES = {
    MASKED: "Mask",
    NON_MASKED: "NonMask",
}

N_ENCODERS = len(ENCODER_DATES)
cmap = mpl.colormaps["managua"]

ENCODER_COLOURS = [colours["masked"], colours["nonmasked"]]

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
