import os

EMBEDDINGS_PATH ="data/output/embeddings"
CLUSTERERS_PATH = "data/output/clusterers"

embeddings = os.listdir(EMBEDDINGS_PATH)
clusterers = os.listdir(CLUSTERERS_PATH)

for index, embedding_filename in enumerate(embeddings):
    if embedding_filename[-1] == "_":
        if embedding_filename[:-1] in embeddings:
            os.remove(
                os.path.join(EMBEDDINGS_PATH, embedding_filename)
            )

for index, clusterer_filename in enumerate(clusterers):
    if clusterer_filename[-1] == "_":
        if clusterer_filename[:-1] in clusterers:
            os.remove(
                os.path.join(CLUSTERERS_PATH, clusterer_filename)
            )


for embedding_filename in embeddings:
    if embedding_filename[:-3] + "pickle" not in clusterers:
        print(embedding_filename)

# for clusterer_filename in clusterers:
#     if clusterer_filename[:-6] + "npy" not in embeddings:
#         print(clusterer_filename)

snapshots_left = []

for snapshot in os.listdir("data/postProcessing/plane"):
    if snapshot + ".npy" not in embeddings:
        snapshots_left.append(snapshot)

print(snapshots_left)
