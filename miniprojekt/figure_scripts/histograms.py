import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import settings

df = pd.read_csv("figure_scripts/data.csv").transpose()

df.drop(["Unnamed: 0", "Benford's Law"], inplace=True)

FIG_X_SCALE = 1
FIG_Y_SCALE = 0.8
DPI = 300

for label in df.index:

    dataset_df = pd.read_csv("data_proc/"+settings.FILES[label]+".csv", header=None)
    #remove zeros:
    dataset_df = dataset_df.loc[(dataset_df != 0).any(axis=1)]

    #Determine bins
    minimum = math.floor(math.log(min(dataset_df[0]), 10))
    maximum = math.floor(math.log(max(dataset_df[0]), 10))
    bins = [10**x for x in range(minimum, maximum+2)]

    fig, ax = plt.subplots(figsize=(FIG_X_SCALE*6.4, FIG_Y_SCALE*4.8), dpi=DPI)
    ax.hist(dataset_df[0], bins=bins)

    ax.set_xlabel("Values in Data Set")
    ax.set_ylabel("Frequency")

    ax.set_xscale("log")
    ax.xaxis.set_ticks(bins)
    plt.minorticks_off()

    fig.tight_layout()
    plt.savefig("figures/hist_"+label.replace(" ", "_") )
    # plt.show()
