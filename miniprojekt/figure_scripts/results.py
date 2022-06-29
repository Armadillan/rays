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
fig, ax = plt.subplots(figsize=(FIG_X_SCALE*6.4, FIG_Y_SCALE*4.8), dpi=DPI)

for label in df.index:
    ax.plot(x, df.loc[label], marker=settings.MARKERS[label])

# ax.set_title("Run 19 average episode lengths")
ax.set_xlabel("Digit")
ax.set_ylabel("Relative Frequency")
ax.legend(df.index, loc="best")
# ax.xaxis.set_ticks(np.arange(min(x), max(x)+1, 2.0))
ax.set_ylim(bottom=0)

fig.tight_layout()

plt.grid(True)

plt.savefig("figures/results.png")

# plt.show()
