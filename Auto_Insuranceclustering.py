# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:47:29 2025

@author: ADMIN
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step A: Load and normalize data
Univ1 = pd.read_csv("C:/4 PDA/AutoInsurance.csv")
Univ = Univ1.drop(columns=["State","Customer","Response","Coverage","Education","Effective To Date","EmploymentStatus",	"Gender",
"Location Code","Marital Status","Policy Type",	
"Policy",	"Renew Offer Type",	"Sales Channel",
"Vehicle Class",	"Vehicle Size"])

def norm_func(df):
    return (df - df.min()) / (df.max() - df.min())

df_norm = norm_func(Univ.iloc[:, 1:])

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
print(Univ.columns)
Univ['clust'] = cluster_labels
Univ1 = Univ.iloc[:, [7, 1, 2, 3, 4, 5, 6]]
print(Univ1.iloc[:, 2:].groupby(Univ1['clust']).mean())

# Step D: Clustering Performance Metrics
silhouette = silhouette_score(df_norm, h_complete.labels_)
db_index = davies_bouldin_score(df_norm, h_complete.labels_)
ch_index = calinski_harabasz_score(df_norm, h_complete.labels_)

print("\n--- Clustering Performance Metrics ---")
print(f"Silhouette Score         : {silhouette:.4f} (range: -1 to +1, higher is better)")
print(f"Davies–Bouldin Index     : {db_index:.4f} (lower is better)")
print(f"Calinski–Harabasz Index  : {ch_index:.4f} (higher is better)")

Univ1.to_csv(".csv", encoding='utf-8')

# Step E: K-Means Elbow Method
Univ1 = pd.read_excel("c:/4 PDA/University_Clustering.xlsx")
Univ1 = Univ1.drop(['State'], axis=1)

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(Univ1.iloc[:, 1:]), columns=Univ1.columns[1:])

TWSS = []
k_range = list(range(2, 8))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_std)
    TWSS.append(kmeans.inertia_)

plt.plot(k_range, TWSS, 'ro-')
plt.xlabel("Number of clusters")
plt.ylabel("Total within sum of squares (TWSS)")
plt.title("Elbow curve to determine optimal k")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Step A: Load and clean data (dropping irrelevant columns)
Univ1 = pd.read_csv("C:/4 PDA/AutoInsurance.csv")
cols_to_drop = ["State","Customer","Response","Coverage","Education",
                "Effective To Date","EmploymentStatus","Gender",
                "Location Code","Marital Status","Policy Type",
                "Policy","Renew Offer Type","Sales Channel",
                "Vehicle Class","Vehicle Size"]
Univ = Univ1.drop(columns=cols_to_drop, errors='ignore')

# Step B: Normalize numeric features
def norm_func(df):
    return (df - df.min()) / (df.max() - df.min())

df_norm = norm_func(Univ.iloc[:, 1:])

# Step C: Plot dendrogram for hierarchical clustering
z = linkage(df_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15, 8))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
dendrogram(z, leaf_rotation=40, leaf_font_size=10)
plt.show()

# Step D: Apply Agglomerative Clustering
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', metric='euclidean').fit(df_norm)
cluster_labels = pd.Series(h_complete.labels_, index=Univ.index)

# Add cluster labels as a new column
Univ['clust'] = cluster_labels
print("Columns in DataFrame:", Univ.columns.tolist())

# Step E: Aggregate statistics by cluster
Univ1_subset = Univ.iloc[:, [7, 1, 2, 3, 4, 5, 6]]
cluster_means = Univ1_subset.iloc[:, 2:].groupby(Univ1_subset['clust']).mean()
print("Cluster-wise feature means:\n", cluster_means)

# Step F: Evaluate clustering performance
silhouette = silhouette_score(df_norm, cluster_labels)
db_index = davies_bouldin_score(df_norm, cluster_labels)
ch_index = calinski_harabasz_score(df_norm, cluster_labels)
print("\n--- Clustering Performance Metrics ---")
print(f"Silhouette Score        : {silhouette:.4f} (higher is better)")
print(f"Davies–Bouldin Index    : {db_index:.4f} (lower is better)")
print(f"Calinski–Harabasz Index : {ch_index:.4f} (higher is better)")

# Step G: Save clustered data if needed
Univ1_subset.to_csv("AutoInsurance.csv", encoding='utf-8', index=False)
