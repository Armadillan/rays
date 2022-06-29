import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import settings

df = pd.read_csv("figure_scripts/data.csv").transpose()

df.drop("Unnamed: 0", inplace=True)

FIG_X_SCALE = 1.3
FIG_Y_SCALE = 1
DPI = 300

x = np.arange(1, 10)
for label in df.index.drop("Benford's Law"):
    fig, ax = plt.subplots(figsize=(FIG_X_SCALE*6.4, FIG_Y_SCALE*4.8), dpi=DPI)
    ax.plot(x, df.loc[label], marker=settings.MARKERS[label])
    ax.plot(x, df.loc["Benford's Law"], marker=settings.MARKERS["Benford's Law"])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Relative Frequency")
    ax.legend([label, "Benford's Law"], loc="best")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("figures/ind_"+label.replace(" ", "_"))
