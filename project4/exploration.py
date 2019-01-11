#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:47:50 2018

@author: leishi
"""

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

import scipy .cluster.hierarchy as sch

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

from plot_functions import plot_silhouette, plot_basis, plot_mds, plot_dendrogram, \
    plot_knn_distance, plot_dbscan, plot_k_elbow, plot_k_with_silhouette

#%%
# Loda data
raw_data = pd.read_csv("Absenteeism_at_work.csv", sep=";")

X = raw_data.values

# Pre-process data before clustering: scale data, (x-mean)/std
X_nor = scale(X)

# Pre-process data before clustering: PCA
X_pca = PCA(n_components=2).fit_transform(X_nor)

# Pre-process data before clustering: 
# Remove ID, Weight, Height
X_removed = np.delete(X_nor, [0, 17, 18], 1)
# PCA after removing ID, Weight, Height
X_pca_removed = PCA(n_components=2).fit_transform(X_removed)

# There are some outliers in the raw dataset
# Later try to run algorithms after removing these outliers


#%%
# K-means approach

# K means with raw data
#%%
# Find optimal number of clusters according sse plot
plot_k_elbow(X, 15)
#%%
# According to plot, choose 2 clusters
k_1 = KMeans(n_clusters=2, init="k-means++", n_init=20, max_iter=300, 
             tol=0.0001, precompute_distances="auto", verbose=0,
             random_state=1, copy_x=True, algorithm="auto")
pred_k_1 = k_1.fit_predict(X)
plot_basis(X, pred_k_1, k_1.cluster_centers_, title="K-means clustering")
print("SEE: ", k_1.inertia_)
print("% instances per cluster: ", np.bincount(pred_k_1)/len(pred_k_1))
#%%
# Try MDS
#k_1_mds = MDS(n_components=2, metric=False, random_state=1).fit_transform(X)
#plt.figure()
#plt.scatter(k_1_mds[:, 0], k_1_mds[:, 1], c=pred_k_1)
#plt.title("MDS")
#plt.show()
plot_mds(X, pred_k_1, title="MDS")

#%%
# Compute silhouette score
k_1_sil_score = silhouette_score(X, pred_k_1)
plot_silhouette(X, pred_k_1)
print("Silhouette Score: ", k_1_sil_score)


#%%
# Find optimal number of clusters according silhouette plot
plot_k_with_silhouette(X, 15)

#%%
# According to plot, choose 2 clusters
#k_2 = KMeans(n_clusters=2, n_init=30, random_state=1)
#pred_k_2 = k_2.fit_predict(X)
#print("SEE: ", k_2.inertia_)
#print("% instances per cluster: ", np.bincount(pred_k_2)/len(pred_k_2))
##%%
#plot_mds(X, pred_k_2)
##%%
#k_2_sil_score = silhouette_score(X, pred_k_2)
#plot_silhouette(X, pred_k_2)
#print("Silhouette Score: ", k_2_sil_score)



#%%
# K means with normalized data
#%%
# Find optimal K
plot_k_elbow(X_nor, 15)
#%%
plot_k_with_silhouette(X_nor, 15)

#%%
# Choose 13 clusters according silhouette score
k_3 = KMeans(n_clusters=13, n_init=20, random_state=1)
pred_k_3 = k_3.fit_predict(X_nor)
print("SEE: ", k_3.inertia_)
print("% instances per cluster: ", np.bincount(pred_k_3)/len(pred_k_3))

#%%
k_3_sil_score = silhouette_score(X_nor, pred_k_3)
plot_silhouette(X_nor, pred_k_3)
print("Silhouette Score: ", k_3_sil_score)
#%%
plot_mds(X_nor, pred_k_3)


#%%
# K means with pca data
#%%
# Find optimal K
plot_k_elbow(X_pca, 15)
#%%
plot_k_with_silhouette(X_pca, 15)

#%%
# choose 4 clusters based elbow
k_4 = KMeans(n_clusters=4, n_init=10, max_iter=500, random_state=1)
pred_k_4 = k_4.fit_predict(X_pca)
print("SEE: ", k_4.inertia_)
print("% instances per cluster: ", np.bincount(pred_k_4)/len(pred_k_4))
#%%
k_4_sil_score = silhouette_score(X_pca, pred_k_4)
plot_silhouette(X_pca, pred_k_4)
print("Silhouette Score: ", k_4_sil_score)
#%%
plot_basis(X_pca, pred_k_4, k_4.cluster_centers_)

#%%
# choose 13 clusters based on silhouette score
k_5 = KMeans(n_clusters=13, n_init=20, random_state=1)
pred_k_5 = k_5.fit_predict(X_pca)
print("SEE: ", k_5.inertia_)
print("% instances per cluster: ", np.bincount(pred_k_5)/len(pred_k_5))
#%%
k_5_sil_score = silhouette_score(X_pca, pred_k_5)
plot_silhouette(X_pca, pred_k_5)
print("Silhouette Score: ", k_5_sil_score)
#%%
plot_basis(X_pca, pred_k_5, k_5.cluster_centers_)


#%%
# Kmeans with removed pca data
#%%
# Find optimal K
plot_k_elbow(X_pca_removed, 15)
plot_k_with_silhouette(X_pca_removed, 15)
#%%
# choose 3 clusters based on elbow
k_6 =KMeans(n_clusters=3, random_state=1)
pred_k_6 = k_6.fit_predict(X_pca_removed)
print("SEE: ", k_5.inertia_)
print("% instances per cluster: ", np.bincount(pred_k_6)/len(pred_k_6))
#%%
k_6_sil_score = silhouette_score(X_pca_removed, pred_k_6)
plot_silhouette(X_pca_removed, pred_k_6)
print("Silhouette Score: ", k_6_sil_score)
#%%
plot_basis(X_pca_removed, pred_k_6, k_6.cluster_centers_)





#%%
# Hierarchy clustering

#%%
h_1 = AgglomerativeClustering(n_clusters=8, affinity="euclidean", 
                              memory=None, connectivity=None,
                              compute_full_tree="auto", linkage="ward")
pred_h_1 = h_1.fit_predict(X)
#%%
#plt.scatter(X[:, 0], X[:, 1], c=pred_h_1)
#plt.title("Hierarchy clustering")
#plt.show()
plot_basis(X, pred_h_1, title="Hierarchy clustering")

#%%
# Plot dendrogram
# =============================================================================
# plt.figure()
# h_1_d = sch.dendrogram(sch.linkage(X, method="ward"))
# plt.show()
# =============================================================================
plot_dendrogram(X, method="single", truncate_mode="level")








#%%
# DBSCAN with raw data
#%%
# Find optimal eps according to customized k using KNN distance
# k corresponds to min_sample
# eps should be the "knee" of knn distance plot




# Try k = 5
plot_knn_distance(X, k=5)
# Based on the plot, choose eps = 30 and min sample = 5
#%%
from datetime import datetime
t1 = datetime.now()
d_1 = DBSCAN(eps=30, min_samples=5, metric="euclidean", algorithm="auto")
pred_d_1 = d_1.fit_predict(X)
t2 = datetime.now()
print("Time taken: ", (t2-t1).microseconds/100000 + (t2 - t1).seconds)
d_1_ls = np.array([x for x in pred_d_1 if x!=-1])
print("% instances per cluster: ", np.bincount(d_1_ls)/len(pred_d_1))
print("% of noise: ", (len(pred_d_1) - len(d_1_ls))/len(pred_d_1))
plot_dbscan(X, d_1)


#%%
# Try k = 20
plot_knn_distance(X, k=20)
#%%
# choose eps = 40, min sample = 20
t1_2 = datetime.now()
d_2 = DBSCAN(eps=40, min_samples=5)
pred_d_2 = d_2.fit_predict(X)
t2_2 = datetime.now()
print("Time taken: ", (t2_2-t1_2).microseconds/100000 + (t2_2 - t1_2).seconds)
d_2_ls = np.array([x for x in pred_d_2 if x!=-1])
print("% instances per cluster: ", np.bincount(d_2_ls)/len(pred_d_2))
print("% of noise: ", (len(pred_d_2) - len(d_2_ls))/len(pred_d_2))
plot_dbscan(X, d_2)


#%%
# DBSCAN with pca data
#%%
# Try k = 10
plot_knn_distance(X_pca, k=10)
#%%
# choose eps=0.4, min sample=10
t1_3 = datetime.now()
d_3 = DBSCAN(eps=0.4, min_samples=10)
pred_d_3 = d_3.fit_predict(X_pca)
t2_3 = datetime.now()
print("Time taken: ", (t2_3-t1_3).microseconds/100000 + (t2_3 - t1_3).seconds)
d_3_ls = np.array([x for x in pred_d_3 if x!=-1])
print("% instances per cluster: ", np.bincount(d_3_ls)/len(pred_d_3))
print("% of noise: ", (len(pred_d_3) - len(d_3_ls))/len(pred_d_3))
plot_dbscan(X, d_3)



#%%
# Hierarchy clustering
h_2 = AgglomerativeClustering(n_clusters=3, affinity="euclidean", 
                              memory=None, connectivity=None,
                              compute_full_tree="auto", linkage="average")
pred_h_2 = h_2.fit_predict(X_pca)
plot_basis(X_pca, pred_h_2)

#%%
# Plot dendrogram
plot_dendrogram(X_pca, method="ward", truncate_mode="lastp")

#%%
plot_dendrogram(X_pca, method="single", truncate_mode="lastp")




#%%
# DBSCAN with normalization
plot_knn_distance(X_nor, k=5)
#%%
# Use eps=3, min sample = 5
d_2 = DBSCAN(eps=3, min_samples=5)
pred_d_2 = d_2.fit_predict(X_nor)
plot_dbscan(X_nor, d_2)
#%%
# DBSCAN with pca
plot_knn_distance(X_pca, k=5)
#%%
d_3 = DBSCAN(eps=0.3, min_samples=5)
pred_d_3 = d_3.fit_predict(X_pca)
plot_dbscan(X_pca, d_3)

