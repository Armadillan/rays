import keyfi as kf
import numpy as np
import os

import pandas as pd

IN_DIR = "data/channel"

OUT_DIR = "data/postProcessing/plane"

os.makedirs(OUT_DIR, exist_ok=True)

dirs = os.listdir(IN_DIR)

length = len(dirs)

for index, dirname in enumerate(dirs):
    filenames = os.listdir(os.path.join(IN_DIR, dirname))
    _, out = kf.import_vtk_data(os.path.join(IN_DIR, dirname, filenames[0]))
    for filename in filenames[1:]:
        _, mesh = kf.import_vtk_data(
            os.path.join(IN_DIR, dirname, filename)
        )
        for var_name in mesh.array_names:
            out[var_name] = mesh.get_array(var_name)
    os.makedirs(
        os.path.join(OUT_DIR, dirname),
        exist_ok=True
    )
    out.save(
        os.path.join(OUT_DIR, dirname, "data.vtk")
    )
    print(f"{(index/length)*100:.2f}%")
