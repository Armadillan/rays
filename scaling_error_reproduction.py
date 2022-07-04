"""
This reproduces the error in the scaling function.
"""

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

#The two methods give slightly different results, WHY?

sc = StandardScaler()
s1 = sc.fit_transform(df)

s2 = df.copy()
s2_mean_ = []
s2_scale_ = []

for label in df.columns:
    scaler = StandardScaler()
    scaler.fit(df[label].values.reshape(-1, 1))
    s2[label] = scaler.transform(df[label].values.reshape(-1,1))
    s2_mean_.append(scaler.mean_)
    s2_scale_.append(scaler.scale_)


#The scale factos are different
#NOT AFTER REINSTALLING SKLEARN???
#BUT THE RESULTS ARE STILL DIFFERENT?
s2_scale_= np.array(s2_scale_).flatten()
# print("scale factors")
# print(sc.scale_)
# print(s2_scale_)

print("scale factor diff")
print(s2_scale_ - sc.scale_)

s2_mean_= np.array(s2_mean_).flatten()
# print("means")
# print(sc.mean_)
# print(s2_mean_)

print("mean diff")
print(s2_mean_ - sc.mean_)

print("scaled data diff description")
print((s2 - s1).describe())

#and when the columns are scaled one by one the mean is much worse
# (further from 0)
print("scaled data descriptions")
print(pd.DataFrame(s1, columns=s2.columns).describe())
print(s2.describe())
