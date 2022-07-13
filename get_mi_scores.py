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

figpath = "figures/correct_all_snapshots"
# vtk_path = "data/postProcessing/correct_all_snapshots"

os.makedirs(CLUSTERERS_PATH, exist_ok=True)
os.makedirs(MI_SCORES_PATH, exist_ok=True)
os.makedirs(figpath, exist_ok=True)

num_embeddings = len(os.listdir(EMBEDDINGS_PATH))

for index, embedding_filename in enumerate(os.listdir(EMBEDDINGS_PATH)):

    start_time=time.time()

    snapshot = embedding_filename[:-4]
    print(f"{index+1}/{num_embeddings}: {snapshot}")

    with open(os.path.join(CLUSTERERS_PATH, snapshot + ".pickle"), "rb") as file:
        clusterer = pickle.load(file)

    embedding = np.load(os.path.join(EMBEDDINGS_PATH, snapshot + ".npy"))

    data, mesh = kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )

    with open(os.path.join(MI_SCORES_PATH, snapshot + ".pickle"), "wb") as file:
        pickle.dump(
            file=file,
            obj= kf.get_cluster_mi_scores(data, clusterer, embedding)
        )

    os.makedirs(os.path.join(vtk_path, snapshot), exist_ok=True)

    kf.export_vtk_data(
        mesh=mesh,
        path=os.path.join(vtk_path, snapshot, "clusters.vtk"),
        cluster_labels=clusterer.labels_
    )

    print(f"{(index+1)*100/num_embeddings:.3f}%")
    print(f"time: {time.time()-start_time}")
