import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import settings

df = pd.read_csv("figure_scripts/error.csv").transpose()

df.drop("Unnamed: 0", inplace=True)

FIG_X_SCALE = 1
FIG_Y_SCALE = 0.8
DPI = 300

fig, ax = plt.subplots(figsize=(FIG_X_SCALE*6.4, FIG_Y_SCALE*4.8), dpi=DPI)
x = np.arange(9)

# plt.grid(True, axis="y")

ax.bar(x, df[0], yerr=df[1], capsize=5)

labels = ["Covid", "Kickstarter", "IKEA", "NEO", "Youtube", "Population", "Budget", "GDP", "Energy"]
ax.set_xticks(x,labels, rotation=-45, ha="left")

ax.set_xlabel("Data Sets")
ax.set_ylabel("Average Deviation from Benford's Law")

fig.tight_layout()

plt.savefig("figures/deviation.png")
