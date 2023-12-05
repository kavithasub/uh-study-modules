# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:34:34 2023

@author: kthat
"""

# ADS - Exercise week 8 - Cluster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster

data = pd.read_csv("C:\\Users\\kthat\\OneDrive\\ADS1\\fish_measurements.csv", skiprows=(1,1))
df = pd.DataFrame(data)
df = df.drop(["species"], axis=1)
#print(df)

#ct.map_corr(df)

df_new = df[["width", "height"]].copy()
#print(df_new)

df_new2, df_min, df_max = ct.scaler(df_new)
print(df_new2)
print(df_min, df_max)

ncluster = 2

plt.figure(figsize=(8, 8))
cm = plt.colormaps["Paired"]
x = df_new2["width"]
y = df_new2["height"]
plt.scatter(x, y, ncluster, marker="d", cmap=cm)
plt.xlabel("species_width")
plt.ylabel("species_height")
plt.show()

kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
kmeans.fit(df_new2)

labels = kmeans.labels_
cen = kmeans.cluster_centers_
#print(cen)