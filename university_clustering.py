# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:44:22 2025

@author: ADMIN
"""

###Clustering:
import pandas as pd
import matplotlib.pylab as plt
Univ1=pd.read_excel("C:/4 PDA/University_Clustering.xlsx")
a=Univ1.describe()

Univ=Univ1.drop(["State"],axis=1)
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(Univ.iloc[:,1:])
b=df_norm.describe()
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical clustering dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")

sch.dendrogram(z,leaf_rotation=40,leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric='euclidean').fit(df_norm)

h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

Univ['clust']=cluster_labels
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()

########
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score 

silhouetee=silhouette_score(df_norm,h_complete.labels_)
db_index=davies_bouldin_score(df_norm,h_complete.labels_)
ch_index=calinski_harabasz_score(df_norm,h_complete.labels_)

print("\n---clustering performance metric---")
print(f"silhouetee score   :{silhouetee:4f}(range:-1 to +1,higher is) ")

print(f"davies-Bouldin index :{db_index:4f}(lower is better)")

print(f"calinski-Harabasz index :{ch_index:.4f} (higher is better)")
      
Univ1.to_csv("University.csv",encoding='utf-8')


####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
Univ1=pd.read_excel("c:/6-p/University_Clustering.xlsx")
Univ1=Univ1.drop(['State'],axis=1)
    
scaler=StandardScaler()
df_std=pd.DataFrame(scaler.fit_transform(Univ1.iloc[:,1:]),columns=Univ1.columns[1:])
TWSS=[]
k_range=list(range(2,8))

for k in k_range:
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(df_std)
    TWSS.append(kmeans.inertia_)
    
plt.plot(k_range,TWSS,'ro-')
plt.xlabel("Number of clusters")
plt.ylabel("Total within sum of squares(TWSS)")
plt.title("Elbow curve to determine optimal k")
plt.show()    

