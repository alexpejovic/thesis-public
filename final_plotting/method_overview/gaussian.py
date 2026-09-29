from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def plot_gaussian(c: Literal["blue", "black", "orange"]):
    # Define grid
    x = np.linspace(-2.8, 2.8, 100)
    y = np.linspace(-2.8, 2.8, 100)
    X, Y = np.meshgrid(x, y)

    # Define mean and covariance
    if c == "black":
        mu = np.array([-0.2, 0.5])
        sigma = np.array([[1.0, 0.2], [0.2, 1.0]])
    elif c == "blue":
        mu = np.array([0.5, -0.2])
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    elif c == "orange":
        mu = np.array([-0.2, 0.5])
        sigma = np.array([[1.0, 0.2], [0.2, 1.0]])

    # Calculate PDF
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mu, sigma)
    Z = rv.pdf(pos)

    # Plot
    with mpl.rc_context(fname="final_plotting/matplotlibrc"):
        fig, ax = plt.subplots(1, 1, figsize=(1, 1), subplot_kw={"projection": "3d"})
        if c == "black":
            ax.plot_surface(X, Y, Z, cmap="Greys", clim=[-0.15, 0.25])
        elif c == "blue":
            ax.plot_surface(X, Y, Z, cmap="Blues", clim=[-0.1, 0.25])
        elif c == "orange":
            ax.plot_surface(X, Y, Z, cmap="Oranges", clim=[-0.1, 0.25])
        ax.set_axis_off()
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        plot_save_path = Path("./final_plotting/method_overview/svg")
        fig.savefig(
            plot_save_path / f"gaussian_{c}.svg",
            bbox_inches="tight",
            pad_inches=-0.075,
        )
        plt.close(fig)


if __name__ == "__main__":
    plot_gaussian("black")
    plot_gaussian("blue")
    plot_gaussian("orange")
