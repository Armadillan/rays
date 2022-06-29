from typing import Sequence, Tuple, Type

import pandas as pd
import pyvista as pv
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from vtk_tools import import_vtk_data

snapshot = 500.5

file_dir = f"data/postProcessing/sphere/{snapshot}/U_zNormal.vtk"

df, mesh = import_vtk_data(file_dir, "U")

sns.scatterplot(data=df, x="U:0", y="U:1", hue="U:2")

plt.show()

sns.pairplot(df)

plt.show()
