# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 23:21:47 2025

@author: ADMIN
"""
'''
2.Perform clustering for the crime data and identify the number of clusters            formed and draw inferences. Refer to crime_data.csv dataset.
'''
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step A: Load and normalize data
Univ1 = pd.read_csv("C:/4 PDA/crime_data.csv")
#Univ = Univ1.drop(["ID"], axis=1)

def norm_func(df):
    return (df - df.min()) / (df.max() - df.min())

df_norm = norm_func(Univ1.iloc[:, 1:])

# Step B: Plot dendrogram
z = linkage(df_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15, 8))
plt.title("Hierarchical clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
dendrogram(z, leaf_rotation=40, leaf_font_size=10)
plt.show()

# Step C: Agglomerative Clustering
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', metric='euclidean').fit(df_norm)
cluster_labels = pd.Series(h_complete.labels_)  # Corrected Series instantiation

Univ1['cluster'] = cluster_labels
Univ_subset = Univ1[["cluster","Assault","UrbanPop","Rape"]]
print(Univ_subset.groupby('cluster').mean())


# Step D: Clustering Performance Metrics
silhouette = silhouette_score(df_norm, h_complete.labels_)
db_index = davies_bouldin_score(df_norm, h_complete.labels_)
ch_index = calinski_harabasz_score(df_norm, h_complete.labels_)

print("\n--- Clustering Performance Metrics ---")
print(f"Silhouette Score         : {silhouette:.4f} (range: -1 to +1, higher is better)")
print(f"Davies–Bouldin Index     : {db_index:.4f} (lower is better)")
print(f"Calinski–Harabasz Index  : {ch_index:.4f} (higher is better)")

Univ_subset.to_csv("C:/4 PDA/crime_clust.csv", encoding='utf-8')

# Step E: K-Means Elbow Method
Univ1 = pd.read_csv("C:/4 PDA/crime_data.csv")
#Univ1 = Univ1.drop(['ID'], axis=1)

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(Univ1.iloc[:, 1:]), columns=Univ1.columns[1:])

TWSS = []
k_range = list(range(2, 6))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_std)
    TWSS.append(kmeans.inertia_)

plt.plot(k_range, TWSS, 'ro-')
plt.xlabel("Number of clusters")
plt.ylabel("Total within sum of squares (TWSS)")
plt.title("Elbow curve to determine optimal k")
plt.show()
