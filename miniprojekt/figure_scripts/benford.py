import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def benford(a):
    return math.log10((a+1)/a)

x = [i for i in range(1, 10)]
data = [benford(i) for i in x]

FIG_X_SCALE = 1
FIG_Y_SCALE = 1
DPI = 300

fig, ax = plt.subplots(figsize=(FIG_X_SCALE*6.4, FIG_Y_SCALE*4.8), dpi=DPI)

ax.bar(x, data)

ax.set_xlabel("Digit")
ax.set_ylabel("Relative Frequency")
ax.xaxis.set_ticks(x)
ax.set_ylim(bottom=0)

fig.tight_layout()

# plt.grid(True)

plt.savefig("figures/benford.png")

# plt.show()
