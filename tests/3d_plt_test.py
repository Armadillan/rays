from typing import Sequence, Tuple, Type

import pandas as pd
import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt

from keyfi import import_vtk_data

snapshot = 500.5

file_dir = f"data/sphere/{snapshot}/U_zNormal.vtk"

df, mesh = import_vtk_data(file_dir)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["U:0"], df["U:1"], df["U:2"], s=0.8, marker=".")

ax.set_xlabel('X Velocity')
ax.set_ylabel('Y Velocity')
ax.set_zlabel('Z Velocity')

plt.show()
