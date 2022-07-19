import os
import time
from io import StringIO
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.cluster import HDBSCAN

DATA_PATH = "data/postProcessing/plane"
EMBEDDINGS_PATH ="data/output/embeddings"
CLUSTERERS_PATH = "data/output/clusterers"
MI_SCORES_PATH = "data/output/mi_scores"

os.makedirs(CLUSTERERS_PATH, exist_ok=True)
os.makedirs(MI_SCORES_PATH, exist_ok=True)

for index, embedding_filename in enumerate(os.listdir(EMBEDDINGS_PATH)):

    start_time=time.time()

    snapshot = embedding_filename[:-4]
    print(f"{index+1}/866: {snapshot}")
    try:
        with open(os.path.join(CLUSTERERS_PATH, snapshot + ".pickle"), "rb") as file:
            clusterer = pickle.load(file)
    except FileNotFoundError:
        continue


    embedding = np.load(os.path.join(EMBEDDINGS_PATH, snapshot + ".npy"))

    data, mesh = kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )
    try:
        with open(os.path.join(MI_SCORES_PATH, snapshot + ".pickle"), "rb") as file:
            snapshot_mi_scores = pickle.load(file)
    except FileNotFoundError:
        continue

    # kf.plot_embedding(embedding)
    kf.plot_cluster_membership(
        embedding=embedding,
        clusterer=clusterer,
        soft=False
        )

    print(f"{(index+1)/8.66:.3f}%")
    print(f"time: {time.time()-start_time}")



snapshot = "285098.078369"

with open(os.path.join(CLUSTERERS_PATH, snapshot + ".pickle"), "rb") as file:
    clusterer = pickle.load(file)

embedding = np.load(os.path.join(EMBEDDINGS_PATH, snapshot + ".npy"))

data, mesh = kf.import_vtk_data(
    os.path.join(DATA_PATH, snapshot, "data.vtk")
)

with open(os.path.join(MI_SCORES_PATH, snapshot + ".pickle"), "rb") as file:
    snapshot_mi_scores = pickle.load(file)

kf.plot_embedding(embedding)
kf.plot_cluster_membership(
    embedding=embedding,
    clusterer=clusterer,
    soft=False
    )

clusterer.generate_prediction_data()

kf.show_condensed_tree(clusterer, label_clusters=False,
    leaf_separation=1,
    log_size=True)
