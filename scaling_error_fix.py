"""
This showcases the "fix" for the error in scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keyfi as kf

from io import StringIO


df1, _ = kf.import_vtk_data("data/postProcessing/plane/285140.078369/data.vtk")


# This is the "fix"
# Save the dataframe as csv and read it back
df = pd.read_csv(StringIO(df1.to_csv()), index_col=0)


# The data is no longe equal
print((df == df1).values.all())

single_scaler = StandardScaler()
scaled_data_1 = pd.DataFrame(
    data=single_scaler.fit_transform(df),
    columns=df.columns)

scaled_data_2 = df.copy()
scaled_data_2_scale_ = []

for column in scaled_data_2.columns:
    fresh_scaler = StandardScaler()
    scaled_data_2[column] = fresh_scaler.fit_transform(df[column].values.reshape(-1, 1))
    scaled_data_2_scale_.append(fresh_scaler.scale_)

#But the scaling is consistent (and good)
print((scaled_data_1 == scaled_data_2).values.all())
