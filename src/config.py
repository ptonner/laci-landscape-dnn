import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import phlanders


def pltcfg():
    plt.style.use("default")

    import seaborn as sns

    sns.set()

    # set global default style:
    sns.set_style("white")
    sns.set_style(
        "ticks",
        {
            "xtick.direction": "in",
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.right": True,
        },
    )
    # sns.set_style({"axes.labelsize": 20, "xtick.labelsize" : 16, "ytick.labelsize" : 16})

    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16

    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["legend.edgecolor"] = "k"


manuscript_labels = {
    "ic50": "$\mathrm{EC}_{50}$",
    "high-level": "$\mathrm{G}_{\infty}$",
    "low-level": "$\mathrm{G}_{0}$",
    "n": "$\mathrm{n}$",
}

data = pd.read_csv("data/median/train.csv")


def unnorm(x, param, shift):

    col = "log-{}-norm".format(param)
    colu = "log-{}".format(param)

    _X = np.ones((data.shape[0], 2))
    _X[:, 0] = data[colu].values
    _b = data[col].values

    _theta, _, _, _ = np.linalg.lstsq(_X, _b, rcond=None)
    sigma = 1 / _theta[0]
    mu = -_theta[1] * sigma

    return (x - shift) * sigma + mu


_experiment = phlanders.experiment.Experiment.load(
    "experiments/final/median/all/cfg.json"
)

shifts = dict(
    [
        (f.feature.name[4:-5], f.shift)
        for f in _experiment.dataset.features["y"].features
    ]
)
