from typing import Sequence, Tuple, Type

import pandas as pd
import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt

# from keyfi import import_vtk_data

from sklearn.datasets import make_blobs

snapshot = 500.5

# file_dir = f"/home/antoni/stuff/Rays/project/data/sphere/{snapshot}/U_zNormal.vtk"

# df, mesh = import_vtk_data(file_dir)
# x = np.linspace(0, 10, 100)
# y = np.sin(x)

data = np.random.randn(2, 100)

fig, ax = plt.subplots()

ax.scatter(*data)
# ax.set_xlim((-0.5, 2))
# ax.set_ylim((-1.5, 1.5))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
plt.show()
