import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

normal_feats = np.load("features/CRL/normal_maxilla_features.npy")

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(normal_feats)

plt.scatter(reduced[:,0], reduced[:,1], alpha=0.6)
plt.title("PCA of Normal Feature Vectors")
plt.show()
