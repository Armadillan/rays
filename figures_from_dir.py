import os
import time
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf

DATA_PATH = "data/postProcessing/plane"
SNAPSHOT = "285140.078369"

TEST_NAME = "drop_Uy_test"

figpath = os.path.join("figures", TEST_NAME)
embedding_path = os.path.join("data/embeddings", TEST_NAME, SNAPSHOT)

os.makedirs(figpath, exist_ok=True)

def get_data(snapshot):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )

df, mesh = get_data(SNAPSHOT)

df = pd.read_csv(StringIO(df.to_csv()), index_col=0)

cleaned_data = kf.clean_data(df, dim=2,
                             vars_to_drop=["N2", "NO2", "rho"]
                            )
for filename in os.listdir(embedding_path):

    embedding = np.load(
        os.path.join(embedding_path, filename)
        )

    for var in ["N2O4", "Qdot", "U:0"]:
        if var == "Qdot":
            if "_200" in filename:
                cmap_minmax=[-200, 200]
            elif "_300" in filename:
                cmap_minmax=[-300, 300]
            else:
                cmap_minmax=[-400, 400]
        else:
            cmap_minmax=[]


        kf.plot_embedding(
            embedding=embedding,
            data=cleaned_data,
            scale_points = True,
            cmap_var=var,
            cmap_minmax=cmap_minmax,
            save=True,
            title=None,
            figname=filename[:-4] + f"_{var}",
            figpath=figpath,
            view=(None, None)
        )
