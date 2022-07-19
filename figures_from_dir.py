import os
import time
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf

DATA_PATH = "data/postProcessing/plane"

TEST_NAME = "drop_Uy_test"

figpath = os.path.join("figures", "ART")
embedding_path = os.path.join("/home/antoni/stuff/Rays/project/data/embeddings/all_snapshots/250_0.1_300c")

os.makedirs(figpath, exist_ok=True)

def get_data(snapshot):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )

for index, filename in enumerate(os.listdir(embedding_path)):

    df, mesh = get_data(filename[:-4])

    embedding = np.load(
        os.path.join(embedding_path, filename)
        )

    for var in ["N2O4", "Qdot", "U:0"]:
        if var == "Qdot":
            cmap_minmax=[-300, 300]
        else:
            cmap_minmax=[]


        kf.plot_embedding(
            embedding=embedding,
            data=df,
            scale_points = True,
            cmap_var=var,
            cmap_minmax=cmap_minmax,
            save=True,
            title=None,
            figname=str(index) + f"_{var}",
            figpath=figpath,
            view=(None, None)
        )
