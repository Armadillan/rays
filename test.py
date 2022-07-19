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
CLUSTERERS_PATH = "data/output/clusterers"
MI_SCORES_PATH = "data/output/mi_scores"
VTK_PATH = "data/postProcessing/correct_all_snapshots"

figpath = "figures/final"

snapshot = "285140.078369"

def get_data(snapshot):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )


df, mesh = get_data(snapshot)

data = kf.clean_data(df, dim=2,
                            vars_to_drop=["N2", "NO2", "rho"]
                        )
data["Qdot"].clip(-300, 300, inplace=True)

#SCALING U:1 AND U:0 SEPERATELY
features_to_scale = ['T', 'Qdot', "N2O4", 'U:0', 'U:1']
scalers = [MaxAbsScaler] * 5

data = kf.scale_data(data, features_to_scale, scalers)

print(f"{snapshot}")

embedding, mapper = kf.embed_data(
    data=data,
    algorithm=UMAP,
    n_neighbors=250,
    min_dist=0.1,
    #reproducible
    # random_state=0,
    n_components=2,
)

kf.plot_embedding(
    embedding=embedding,
    data=data,
    # figsize=(7.2, 7.2),
    scale_points=0.6,
    cmap_var="U:1",
    # cmap_minmax=[-300, 300],
    save=True,
    figpath=figpath,
    figname="jellyfish.png"
    )
