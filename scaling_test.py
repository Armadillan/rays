import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.dimred import UMAP
from keyfi.cluster import HDBSCAN

from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/postProcessing/plane"

def get_data(path):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, path)
    )

df, mesh = get_data("285140.078369/data.vtk")

cleaned_data = kf.clean_data(df, dim=2, vars_to_drop=["N2"])

#The two methods give slightly different results, WHY?

sc = StandardScaler()
s1 = sc.fit_transform(cleaned_data)

s2 = cleaned_data.copy()
s2_mean_ = []
s2_scale_ = []

for label in cleaned_data.columns:
    scaler = StandardScaler()
    scaler.fit(cleaned_data[label].values.reshape(-1, 1))
    s2[label] = scaler.transform(cleaned_data[label].values.reshape(-1,1))
    s2_mean_.append(scaler.mean_)
    s2_scale_.append(scaler.scale_)

s2_mean_= np.array(s2_mean_)
s2_scale_= np.array(s2_scale_)
# print(s2_mean_)
# print(sc.mean_)
print("scale factors")
print(sc.scale_)
print(s2_scale_.flatten())

print("scale factor diff")
print(s2_scale_.flatten() - sc.scale_)

print("scaled data diff df")
print((s2 - s1).describe())

print("scaled data descriptions")
print(pd.DataFrame(s1, columns=s2.columns).describe())
print(s2.describe())
