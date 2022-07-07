import os
import time
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.cluster import HDBSCAN


DATA_PATH = "data/postProcessing/plane"
SNAPSHOT = "285140.078369"

def get_data(snapshot):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )


embedding_path = os.path.join(
    "data/embeddings/final_umap_test",
    SNAPSHOT,
    "250_0.1_300c.npy"
    )

embedding = np.load(embedding_path)

df, mesh = get_data(SNAPSHOT)

clusterer = kf.cluster_embedding(
    embedding=embedding,
    algorithm=HDBSCAN,
    min_cluster_size=300,
    min_samples=10,
    prediction_data=True,
)

# kf.plot_embedding(
#     embedding=embedding,
#     data=df,
#     scale_points = True,
#     cmap_var="N2O4",
#     cmap_minmax=[],
# )

for index in range(49409):
    if -9.4 <= embedding[index][0] <= 8:
        if -7.7 <= embedding[index][1] <= 1.3:
            clusterer.labels_[index] = 2

kf.plot_cluster_membership(
    embedding=embedding,
    clusterer=clusterer,
    soft=False
)

for i in range(3):
    kf.get_cluster_mi_scores(
        data=df,
        clusterer=clusterer,
        embedding=embedding,
        cluster_num = i,
        scale = False,
        flag_print = False,
        flag_plot = True
    )

path_output = 'data/manual_clusters.vtk'
kf.export_vtk_data(mesh=mesh, path=path_output, cluster_labels=clusterer.labels_)
