import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

normal_feats = np.load("features/CRL/normal_abdomen_features.npy")

# PCA
pca = PCA()
reduced = pca.fit_transform(normal_feats)

plt.scatter(reduced[:,0], reduced[:,1], alpha=0.6)
plt.title("PCA of Normal Feature Vectors")
plt.show()
