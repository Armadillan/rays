#This has been superseded by foamSequenceVTKFiles

import os
import re
import shutil

regex = re.compile(r"\.(0)+")

DIR = "data/sphere"
VAR = "U"
ODIR = "o_data"

os.makedirs(ODIR, exist_ok=True)

snapshots = sorted([float(x) for x in os.listdir(DIR)])
snapshots = [str(x) for x in snapshots]
snapshots = [regex.sub("", x) for x in snapshots]

num_snapshots = len(snapshots)

for i, dirname in enumerate(snapshots):
    os.link(
        os.path.join(DIR,dirname, f"{VAR}_zNormal.vtk"),
        os.path.join(ODIR, f"{DIR}_{VAR}_{i}" + ".vtk"))
    print((i+1)/num_snapshots)
