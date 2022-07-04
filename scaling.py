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

from sklearn.preprocessing import StandardScaler, MaxAbsScaler

DATA_PATH = "data/postProcessing/plane"

def get_data(path):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, path)
    )


df, mesh = get_data("285140.078369/data.vtk")

df = pd.read_csv(StringIO(df.to_csv()), index_col=0)

#should U_z be dropped? dim=3 does not drop U_z
cleaned_data = kf.clean_data(df, dim=2, vars_to_drop=["N2", "NO2", "rho"])

print(cleaned_data.columns)

cleaned_data["Qdot"].clip(-200, 200, inplace=True)

features = [feature for feature in cleaned_data.columns[:-2]] + [["U:0", "U:1"]]
standard_scalers = ([StandardScaler] * len(cleaned_data.columns[:-2])) + [StandardScaler]
maxabs_scalers = ([MaxAbsScaler] * len(cleaned_data.columns[:-2])) + [MaxAbsScaler]

print(features)
print(standard_scalers)
print(maxabs_scalers)

standard = kf.scale_data(cleaned_data, features, standard_scalers)

maxabs = kf.scale_data(cleaned_data, features, maxabs_scalers)

BINS = 200

fig, axes = plt.subplots(3, 5)

axes[2,0].hist(cleaned_data["T"], bins=BINS)
axes[2,0].set_title("T, not scaled")
axes[2,1].hist(cleaned_data["N2O4"], bins=BINS)
axes[2,1].set_title("N2O4, not scaled")
axes[2,2].hist(cleaned_data["Qdot"], bins=BINS)
axes[2,2].set_title("Qdot, not scaled")
axes[2,3].hist(cleaned_data["U:0"], bins=BINS)
axes[2,3].set_title("U:0, not scaled")
axes[2,4].hist(cleaned_data["U:1"], bins=BINS)
axes[2,4].set_title("U:1, not scaled")

axes[0,0].hist(standard["T"], bins=BINS)
axes[0,0].set_title("T, Standard")
axes[0,1].hist(standard["N2O4"], bins=BINS)
axes[0,1].set_title("N2O4, Standard")
axes[0,2].hist(standard["Qdot"], bins=BINS)
axes[0,2].set_title("Qdot, Standard")
axes[0,3].hist(standard["U:0"], bins=BINS)
axes[0,3].set_title("U:0, Standard")
axes[0,4].hist(standard["U:1"], bins=BINS)
axes[0,4].set_title("U:1, Standard")

axes[1,0].hist(maxabs["T"], bins=BINS)
axes[1,0].set_title("T, MaxAbs")
axes[1,1].hist(maxabs["N2O4"], bins=BINS)
axes[1,1].set_title("N2O4, MaxAbs")
axes[1,2].hist(maxabs["Qdot"], bins=BINS)
axes[1,2].set_title("Qdot, MaxAbs")
axes[1,3].hist(maxabs["U:0"], bins=BINS)
axes[1,3].set_title("U:0, MaxAbs")
axes[1,4].hist(maxabs["U:1"], bins=BINS)
axes[1,4].set_title("U:1, MaxAbs")

plt.show()
