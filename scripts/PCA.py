import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns
df = pd.read_csv(r"../data/fashion-mnist_test.csv")
print(df.shape)


# visualize 3x3 images
imgs = df.iloc[:, 1:].values
imgs = imgs.reshape(-1, 28, 28)
labels = df.iloc[:, 0]

fig, ax = plt.subplots(3, 3, figsize=(12, 12))
for i in range(3):
    for j in range(3):
        img = random.choice(imgs)
        ax[i][j].imshow(img, cmap='gray')

plt.show()

pca = PCA(n_components=2)
pca.fit(imgs.reshape(10000, -1))

print(f"explained variance ratio: {pca.explained_variance_ratio}")
print(f"singular values: {pca.singular_values_}")

imgs_reduced = pca.transform(imgs.reshape(10000, -1))
#imgs_reduced =

color_labels = labels.unique()
rgb_values = sns.color_palette("Set2", 10)
color_map = dict(zip(color_labels, rgb_values))
plt.scatter(imgs_reduced[:, 0], imgs_reduced[:, 1], c=labels.map(color_map))

for i in range(color_labels):
    idx = labels == i
    plt.scatter(imgs_reduced[idx, 0], imgs_reduced[idx, 1])

print("finished")