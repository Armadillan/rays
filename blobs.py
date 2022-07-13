from sklearn.datasets import make_blobs

import keyfi as kf
from keyfi.cluster import HDBSCAN

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

points, labels = make_blobs(cluster_std=0.8, n_samples=300)

clusterer = kf.cluster_embedding(
    embedding=points,
    algorithm=HDBSCAN,
    min_cluster_size=10,
    min_samples=10,
)


kf.plot_cluster_membership(
    embedding=points,
    clusterer=clusterer,
    soft=False,
    legend=True,
)
