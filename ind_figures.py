import os
import time
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.cluster import HDBSCAN

# SNAPSHOT = "285864.578369" #555
SNAPSHOT="285063.578369" #21
# SNAPSHOT = "285156.578369" #84
# SNAPSHOT = "285417.578369" # 257


DATA_PATH = "data/postProcessing/plane"
EMBEDDINGS_PATH ="data/output/embeddings"
CLUSTERERS_PATH = "data/output/clusterers"
MI_SCORES_PATH = "data/output/mi_scores"

figpath = "figures/final"

embedding_path = os.path.join("data/output/embeddings", SNAPSHOT + ".npy")

with open(os.path.join(CLUSTERERS_PATH, SNAPSHOT + ".pickle"), "rb") as file:
    clusterer = pickle.load(file)

embedding = np.load(os.path.join(EMBEDDINGS_PATH, SNAPSHOT + ".npy"))

data, mesh = kf.import_vtk_data(
    os.path.join(DATA_PATH, SNAPSHOT, "data.vtk")
)

kf.plot_embedding(
    embedding,
    data=data,
    figsize=(7.2, 7.2),
    scale_points=0.5,
    cmap_var="T",
    #label="$\dot{Q}$", #"$\mathrm{N}_2\mathrm{O}_4$",
    # cmap_minmax=[-300, 300],
    save=True,
    figpath=figpath,
    figname=SNAPSHOT + "t" + ".png"
    )

# kf.plot_cluster_membership(
#     embedding=embedding,
#     clusterer=clusterer,
#     soft=False,
#     legend=True,
#     save=True,
#     figpath=figpath,
#     figname=SNAPSHOT + "clusters" + ".png"
#     )

# clusterer.generate_prediction_data()

# kf.show_condensed_tree(clusterer, select_clusters=True, label_clusters=False)

# for i in (1, 2):#np.unique(clusterer.labels_):
#     print(i)
#     kf.get_cluster_mi_scores(
#         data=data,
#         clusterer=clusterer,
#         embedding=embedding,
#         cluster_num = i,
#         scale = False,
#         flag_print = False,
#         flag_plot = True
#     )
