import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.dimred import UMAP
from keyfi.cluster import HDBSCAN

DATA_PATH = "data/postProcessing/plane"

def get_data(path):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, path)
    )

df, mesh = get_data("285002.078369/data.vtk")

#should U_z be dropped? dim=3 does not drop U_z
cleaned_data = kf.clean_data(df, dim=2, vars_to_drop=None)

variables = cleaned_data.columns

print("Starting embedding")

#UMAP
figpath="figures/UMAP"
os.makedirs(figpath, exist_ok=True)

n_neighbors_range = [50, 100, 150, 200, 250, 300]
min_dist_range = [0.05, 0.1]

for n_neighbors in n_neighbors_range:
    for min_dist in min_dist_range:

        start_time = time.time()

        print(n_neighbors, min_dist)

        embedding, mapper = kf.embed_data(
            data=cleaned_data,
            algorithm=UMAP,
            scale=True,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            #ensures reproducibility, disable for faster compute
            # random_state=0,
            #how many dimensions to reduce to
            n_components=2
        )

        print("saving vars")

        for var in variables:

            if var == "Qdot":
                cmap_minmax=(-300, 300)
            else:
                cmap_minmax=[]

            kf.plot_embedding(
                embedding=embedding,
                data=cleaned_data,
                scale_points = True,
                cmap_var=var,
                cmap_minmax=cmap_minmax,
                save=True,
                title=f"{n_neighbors}_{min_dist}_{var}",
                figname=f"{n_neighbors}_{min_dist}_{var}",
                figpath=figpath
            )

        print(f"time: {(time.time()-start_time):.2f} s")
