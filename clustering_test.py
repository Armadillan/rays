import os
import time
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.cluster import HDBSCAN, KMeans


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

# kf.plot_embedding(
#     embedding=embedding,
#     data=df,
#     scale_points = True,
#     cmap_var="N2O4",
#     cmap_minmax=[],
# )


clusterer = kf.cluster_embedding(
    embedding=embedding,
    algorithm=HDBSCAN,
    min_cluster_size=300,
    min_samples=10,
)

import pickle

with open("clusterer.pickle", "wb") as file:
    pickle.dump(clusterer, file)

with open("clusterer.pickle", "rb") as file:
    clusterer = pickle.load(file)

# kf.plot_cluster_membership(
#     embedding=embedding,
#     clusterer=clusterer,
#     soft=False
# )

# what is this supposed to be?
# kf.plot_clustering(embedding=embedding, cluster_labels=clusterer.labels_)


# kf.show_condensed_tree(
#     clusterer,
#     select_clusters=True,
#     label_clusters=False,
#     leaf_separation=0.2,
#     log_size=False,
# )

cluster_mi_scores = kf.get_cluster_mi_scores(
    data=df,
    clusterer=clusterer,
    embedding=embedding,
    cluster_num = 0,
    scale = False,
    # flag_print = True,
    # flag_plot = True
)

cluster_mi_scores.to_pickle("mo_scores.pickle")

cluster_mi_scores = pd.read_pickle("mo_scores.pickle")

# kf.mi.plot_cluster_mi_scores(cluster_mi_scores)

path_output = 'data/clusters_umap_hdbscan.vtk'
kf.export_vtk_data(mesh=mesh, path=path_output, cluster_labels=clusterer.labels_)
