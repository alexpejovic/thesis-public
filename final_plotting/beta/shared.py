import matplotlib as mpl

ENCODER_DATES = [
    "gelu-beta0",
    # "gelu-beta0.001",
    # "gelu-beta0.02",
    # "gelu-beta0.05",
    # "gelu-beta0.1",
    "gelu-beta0.5",
    # "gelu-beta1.0",
    "gelu-beta2.0",
    "gelu-beta5.0",
    # "gelu-beta10.0",
]

ENCODER_NAMES = {
    "gelu-beta0": r"$\beta = 0$",
    # "gelu-beta0.1": "Beta = 0.1",
    "gelu-beta0.5": r"$\beta = 0.5$",
    # "gelu-beta1.0": "Beta = 1.0",
    "gelu-beta2.0": r"$\beta = 2.0$",
    "gelu-beta5.0": r"$\beta = 5.0$",
    # "gelu-beta10.0": "Beta = 10.0",
}

N_ENCODERS = len(ENCODER_DATES)

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


cmap = mpl.colormaps["managua"]

ENCODER_COLOURS = [cmap((i + 0.3) / N_ENCODERS) for i in range(N_ENCODERS)]

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
    "Leak_gLeak": r"$g_{Leak}$",
    "Leak_eLeak": r"$e_{Leak}$",
    "Na_gNa": r"$g_{Na}$",
    "K_gK": r"$g_{K}$",
    "Km_gKm": r"$g_{Km}$",
    "CaL_gCaL": r"$g_{CaL}$",
    "Km_taumax": r"$\tau_{max}$",
    "vt": r"$V_T$",
}
