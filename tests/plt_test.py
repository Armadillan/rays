from typing import Sequence, Tuple, Type

import pandas as pd
import pyvista as pv
import numpy as np

import matplotlib.pyplot as plt

from vtk_tools import import_vtk_data

snapshot = 500.5

file_dir = f"data/postProcessing/sphere/{snapshot}/U_zNormal.vtk

df, mesh = import_vtk_data(file_dir, "U")

fig, ax = plt.subplots()

ax.scatter(df["U:0"], df["U:1"], 0.8, marker=".")
ax.set_xlim((-0.5, 2))
ax.set_ylim((-1.5, 1.5))
ax.set_xlabel("X Velocity (U:0)")
ax.set_ylabel("Y Velocity (U:1)")
ax.grid(True)
plt.show()
