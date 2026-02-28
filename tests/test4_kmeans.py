import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_energy
from models.kmeans import KMeansScratch
import numpy as np

print("Loading Energy dataset...")
X, y = load_energy()

print("\nRunning K-Means with k=3...")
km = KMeansScratch(k=3)
km.fit(X)

print("\nK-Means Results:")
print("Centroids shape:", km.centroids.shape)
print("Labels shape:", km.labels.shape)
print("Unique labels:", set(km.labels))
print("Cluster sizes:", [np.sum(km.labels == i) for i in range(3)])
