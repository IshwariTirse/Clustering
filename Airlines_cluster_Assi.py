# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 22:21:38 2025

@author: ADMIN
"""

import pandas as pd
#data handling (DataFrames).
import matplotlib.pyplot as plt
#ploting
from scipy.cluster.hierarchy import linkage, dendrogram
#functions to build/plot hierarchical clustering trees.
from sklearn.cluster import AgglomerativeClustering
#scikit-learn’s hierarchical clustering model.
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
#clustering quality metrics.
from sklearn.cluster import KMeans
#k-means algorithm.
from sklearn.preprocessing import StandardScaler
#standardize features (mean 0, std 1).
# Step A: Load and normalize data
Univ1 = pd.read_excel("C:/4 PDA/Airlines.xlsx")
#read the Excel file into a DataFrame.
Univ = Univ1.copy()
#remove the ID column (identifier; not useful for clustering).

def norm_func(df):
    return (df - df.min()) / (df.max() - df.min())
#min–max scaling to 0–1 range.
df_norm = norm_func(Univ.iloc[:, 1:])
#apply min–max scaling to all columns except the first one of Univ.
# Step B: Plot dendrogram
z = linkage(df_norm, method="complete", metric="euclidean")
#compute hierarchical linkage using complete linkage (clusters’ distance = max pairwise distance) and Euclidean metric.
plt.figure(figsize=(15, 8))
#big figure.Titles/labels.
plt.title("Hierarchical clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
dendrogram(z, leaf_rotation=40, leaf_font_size=10)
plt.show()

# Step C: Agglomerative Clustering
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', metric='euclidean').fit(df_norm)
cluster_labels = pd.Series(h_complete.labels_)  # Corrected Series instantiation
#fit hierarchical clustering, asking for 3 clusters 
#with the same settings as the dendrogram.
Univ['cc1_miles'] = cluster_labels
Univ_subset = Univ[["Balance","cc1_miles","Bonus_miles","Bonus_trans",	"Flight_miles_12mo","Flight_trans_12","Days_since_enroll"]]
print(Univ_subset.groupby('cc1_miles').mean())
#keep a few key columns plus the cluster label for profiling.

# Step D: Clustering Performance Metrics
silhouette = silhouette_score(df_norm, h_complete.labels_)
#Silhouette (−1 to +1; higher is better).
db_index = davies_bouldin_score(df_norm, h_complete.labels_)
#Davies–Bouldin (lower is better).
ch_index = calinski_harabasz_score(df_norm, h_complete.labels_)
#Calinski–Harabasz (higher is better).
print("\n--- Clustering Performance Metrics ---")
print(f"Silhouette Score         : {silhouette:.4f} (range: -1 to +1, higher is better)")
print(f"Davies–Bouldin Index     : {db_index:.4f} (lower is better)")
print(f"Calinski–Harabasz Index  : {ch_index:.4f} (higher is better)")

Univ_subset.to_csv("C:/4 PDA/Airline_clust.csv", encoding='utf-8')
#reload raw data (fresh copy).
# Step E: K-Means Elbow Method
Univ1 = pd.read_excel("C:/4 PDA/Airlines.xlsx")
Univ1 = Univ1.drop(['ID'], axis=1)

scaler = StandardScaler()
#create a standardizer.
#df_std = pd.DataFrame(scaler.fit_transform(Univ1.iloc[:, 1:]), columns=Univ1.columns[1:])
df_std = pd.DataFrame(scaler.fit_transform(Univ1), columns=Univ1.columns)

#standardize all columns except the first one (again, first feature is excluded).
TWSS = []
#list for Total Within Sum of Squares (k-means inertia).
k_range = list(range(2, 12))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_std)
    TWSS.append(kmeans.inertia_)

plt.plot(k_range, TWSS, 'ro-')
plt.xlabel("Number of clusters")
plt.ylabel("Total within sum of squares (TWSS)")
plt.title("Elbow curve to determine optimal k")
plt.show()
