import os
import time
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.dimred import UMAP
from keyfi.cluster import HDBSCAN

from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

DATA_PATH = "data/postProcessing/plane"

def get_data(path):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, path)
    )

df, mesh = get_data("285140.078369/data.vtk")

#should U_z be dropped? dim=3 does not drop U_z
cleaned_data = kf.clean_data(df, dim=2, vars_to_drop=["N2"])

fig, axes = plt.subplots(2, 2)

#"fix"
qdot = pd.read_csv(StringIO(cleaned_data.to_csv()), index_col=0)["Qdot"]

clipped200 = qdot.clip(-200, 200)

clipped100 = qdot.clip(-100, 100)

BINS = 100

axes[0, 0].hist(
    MaxAbsScaler().fit_transform(clipped200.values.reshape(-1,1)),
    bins=BINS)

axes[0, 0].set_title("200, scaled")

axes[0, 1].hist(
    clipped200,
    bins=BINS
)
axes[0, 1].set_title("200")

axes[1, 0].hist(
    MaxAbsScaler().fit_transform(clipped100.values.reshape(-1,1)),
    bins=BINS)

axes[1, 0].set_title("100, scaled")

axes[1, 1].hist(
    clipped100,
    bins=BINS
)
axes[1, 1].set_title("100")

plt.show()

plt.hist(df["Qdot"], bins=100)
plt.tight_layout()
plt.xlabel("$\dot{Q}$")
plt.ylabel("Frequency")
plt.savefig("figures/final/Qdot_nonclipped.png", bbox_inches='tight')
plt.show()

plt.hist(df["Qdot"].clip(-300, 300), bins=100)
plt.tight_layout()
plt.xlabel("$\dot{Q}$")
plt.ylabel("Frequency")
plt.savefig("figures/final/Qdot_300clipped.png", bbox_inches='tight')
plt.show()
