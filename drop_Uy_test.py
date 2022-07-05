import os
import time
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keyfi as kf
from keyfi.dimred import UMAP
from keyfi.cluster import HDBSCAN

from sklearn.preprocessing import StandardScaler, MaxAbsScaler

DATA_PATH = "data/postProcessing/plane"
SNAPSHOT = "285140.078369"

#savepaths
figpath="figures/drop_Uy_test"
embedding_path = os.path.join("data/embeddings/drop_Uy_test", SNAPSHOT)

os.makedirs(figpath, exist_ok=True)
os.makedirs(embedding_path, exist_ok=True)

def get_data(snapshot):
    return kf.import_vtk_data(
        os.path.join(DATA_PATH, snapshot, "data.vtk")
    )

df, mesh = get_data(SNAPSHOT)

df = pd.read_csv(StringIO(df.to_csv()), index_col=0)

cleaned_data = kf.clean_data(df, dim=2,
                             vars_to_drop=["N2", "NO2", "rho", "U:1"]
                            )

data200 = cleaned_data.copy()
data200["Qdot"].clip(-200, 200, inplace=True)

data300 = cleaned_data.copy()
data300["Qdot"].clip(-300, 300, inplace=True)

data400 = cleaned_data.copy()
data400["Qdot"].clip(-400, 400, inplace=True)

features_U = cleaned_data.columns
features_no_U = cleaned_data.columns[:-1]

scalers_U = [MaxAbsScaler] * 4
scalers_no_U = [MaxAbsScaler] * 3

data200U = kf.scale_data(data200, features_U, scalers_U)
data300U = kf.scale_data(data300, features_U, scalers_U)
data400U = kf.scale_data(data400, features_U, scalers_U)

data200n = kf.scale_data(data200, features_no_U, scalers_no_U)
data300n = kf.scale_data(data300, features_no_U, scalers_no_U)
data400n = kf.scale_data(data400, features_no_U, scalers_no_U)

saved_embeddings = os.listdir(embedding_path)

for n_neighbors in (300, 200):
    for data, description in zip(
        (data200U, data300U, data400U, data200n, data300n, data400n),
        ("data200U", "data300U", "data400U", "data200n", "data300n", "data400n")):

        start_time = time.time()

        print(n_neighbors, 0.1, description)

        if f"{n_neighbors}_0.1_{description}.npy" in saved_embeddings:
            embedding = np.load(
                os.path.join(embedding_path, f"{n_neighbors}_0.1_{description}.npy")
                )
            print("loaded from existing")

        else:

            embedding, mapper = kf.embed_data(
                data=data,
                algorithm=UMAP,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                #ensures reproducibility, disable for faster compute
                random_state=0,
                #how many dimensions to reduce to
                n_components=2,
            )

            np.save(
                os.path.join(embedding_path, f"{n_neighbors}_0.1_{description}"),
                embedding
            )

        print("time:", time.time()-start_time)
