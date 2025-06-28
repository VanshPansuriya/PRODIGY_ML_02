import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("Mall_Customers.csv")


data = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Use the Elbow Method to find optimal number of clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Fit KMeans with optimal k (assume k=5 after Elbow method)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)


df['Cluster'] = clusters

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
for i in range(optimal_k):
    plt.scatter(
        reduced_data[clusters == i, 0],
        reduced_data[clusters == i, 1],
        label=f'Cluster {i}'
    )

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Customer Segments (via PCA)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
