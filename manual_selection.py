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
# SNAPSHOT = "285044.078369" #8
SNAPSHOT = "285140.078369" #72

def get_data(snapshot):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )


embedding_path = os.path.join("data/output/embeddings", SNAPSHOT + ".npy")

embedding = np.load(embedding_path)

df, mesh = get_data(SNAPSHOT)

clusterer = kf.cluster_embedding(
    embedding=embedding,
    algorithm=HDBSCAN,
    min_cluster_size=300,
    min_samples=10,
    prediction_data=True,
)

# plt.scatter(*embedding.T, s=0.1)
# plt.show()
# kf.plot_embedding(
#     embedding=embedding,
#     data=df,
#     scale_points = True,
#     cmap_var="N2O4",
#     cmap_minmax=[],
# )

# kf.show_condensed_tree(clusterer, select_clusters=True, label_clusters=False)

for index in range(49409):

    # #8
    # if 17.2 <= embedding[index][0]:
    #     # if -7.7 <= embedding[index][1] <= 1.3:
    #         clusterer.labels_[index] = 5
    # if (embedding[index][1] <= 8.2 and embedding[index][0] >= 3.9) or embedding[index][1] <= 1.5:
    #     clusterer.labels_[index] = 6


    # #72
    if -9.4 <= embedding[index][0] <= 8:
        if -7 <= embedding[index][1] <= -0.5:
            clusterer.labels_[index] = 2
    if -8.9 <= embedding[index][0] <= -7.5:
        clusterer.labels_[index] = 2
    if 13.9 <= embedding[index][0]:
        clusterer.labels_[index] = 3

os.makedirs("figures/manual_clustering", exist_ok=True)

kf.plot_cluster_membership(
    embedding=embedding,
    clusterer=clusterer,
    soft=False,
    save=True,
    figpath="figures/manual_clustering",
    figname=SNAPSHOT+".png"
)
for i in np.unique(clusterer.labels_):
    print(i)
    kf.get_cluster_mi_scores(
        data=df,
        clusterer=clusterer,
        embedding=embedding,
        cluster_num = i,
        scale = False,
        flag_print = False,
        flag_plot = True,
    )

# path_output = 'data/manual_clusters/' + SNAPSHOT + '.vtk'
# kf.export_vtk_data(mesh=mesh, path=path_output, cluster_labels=clusterer.labels_)
