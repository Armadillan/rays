import os
import time
from io import StringIO
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.dimred import UMAP
from keyfi.cluster import HDBSCAN

from sklearn.preprocessing import MaxAbsScaler

DATA_PATH = "data/postProcessing/plane"
EMBEDDINGS_PATH ="data/output/embeddings"
CLUSTERERS_PATH = "data/clusterers"
MI_SCORES_PATH = "data/mi_scores"
logfile = "data/log.txt"

os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
os.makedirs(CLUSTERERS_PATH, exist_ok=True)
os.makedirs(MI_SCORES_PATH, exist_ok=True)

saved_embeddings = os.listdir(EMBEDDINGS_PATH)

def get_data(snapshot):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )

def log(*msg):
    with open(logfile, "a") as file:
        print(*msg, file=file)

features_to_scale = ['T', 'Qdot', "N2O4", ['U:0', 'U:1']]
scalers = [MaxAbsScaler] * 4

#
to_compute = ['285473.078369', '286074.578369', '285425.078369', '285470.078369', '286149.578369', '286118.078369', '285563.078369', '286185.578369', '285483.578369', '285240.578369', '285602.078369', '285083.078369', '285567.578369', '286008.578369', '285911.078369', '286086.578369', '285144.578369', '285116.078369', '285479.078369', '285146.078369', '285047.078369', '285750.578369', '285647.078369', '285213.578369']
num_to_compute = len(to_compute)


for index, snapshot in enumerate(to_compute):

    start_time = time.time()
    #data prep
    df, mesh = get_data(snapshot)

    print(df.describe())
    exit()

    data = kf.clean_data(df, dim=2,
                                vars_to_drop=["N2", "NO2", "rho"]
                            )
    data["Qdot"].clip(-300, 300, inplace=True)

    data = kf.scale_data(data, features_to_scale, scalers)

    log(f"{index+1}/{num_to_compute}: {snapshot}")
    print(f"{index+1}/{num_to_compute}: {snapshot}")

    if f"{snapshot}.npy" in saved_embeddings:
        embedding = np.load(
            os.path.join(EMBEDDINGS_PATH, f"{snapshot}.npy")
            )
        log("loaded from existing")
        print("loaded from existing")

    else:
        embedding, mapper = kf.embed_data(
            data=data,
            algorithm=UMAP,
            n_neighbors=250,
            min_dist=0.1,
            #reproducible
            random_state=0,
            n_components=2,
        )

        np.save(
            os.path.join(EMBEDDINGS_PATH, f"{snapshot}"),
            embedding
        )
    print(f"{(index+1)*100/num_to_compute:.3f}%")
    print(f"time: {time.time()-start_time}")
    log(f"{(index+1)*100/num_to_compute:.3f}%")
    log(f"time: {time.time()-start_time}")

# #HDBSCAN
for index, snapshot_embedding in enumerate(sorted(os.listdir(EMBEDDINGS_PATH))):
    snapshot = snapshot_embedding[:-4]

    embedding = np.load(os.path.join(EMBEDDINGS_PATH, snapshot_embedding))

    clusterer = kf.cluster_embedding(
    embedding=embedding,
    algorithm=HDBSCAN,
    min_cluster_size=300,
    min_samples=10,
    )

    with open(os.path.join(CLUSTERERS_PATH, snapshot + ".pickle"), "wb") as file:
        pickle.dump(clusterer, file)

    print(f"{(index + 1) * 100 /  866:.2f}%")

for index, clusterer_filename in enumerate(os.listdir(CLUSTERERS_PATH)):

    start_time = time.time()

    snapshot = clusterer_filename[:-7]
    print(f"{index+1}/{num_to_compute}: {snapshot}")

    snapshot_mi_scores = {}

    with open(os.path.join(CLUSTERERS_PATH, clusterer_filename), "rb") as file:
        clusterer = pickle.load(file)

    embedding = np.load(os.path.join(EMBEDDINGS_PATH, snapshot + ".npy"))

    data, _ = kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )

    for cluster in np.unique(clusterer.labels_):
        snapshot_mi_scores[cluster] = kf.get_cluster_mi_scores(
            data=data,
            clusterer=clusterer,
            embedding=embedding,
            cluster_num=cluster,
            scale=False,
            flag_print=False,
            flag_plot = False,
        )

    with open(os.path.join(MI_SCORES_PATH, snapshot + ".pickle"), "wb") as file:
        pickle.dump(snapshot_mi_scores, file)

    print(f"{(index+1)*100/num_to_compute:.3f}%")
    print(f"time: {time.time()-start_time}")
