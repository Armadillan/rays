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

data = np.random.randn(3, 100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*data)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
