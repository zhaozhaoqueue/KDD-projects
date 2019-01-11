#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:29:45 2018

@author: leishi
"""
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import scipy .cluster.hierarchy as sch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_silhouette(X, cluster_labels):
    n_clusters = len(set(cluster_labels))
    # Compute silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    # Compute silhouette value for each data point
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
#    fig, ax = plt.subplot()
#    fig = plt.figure()
    ax = plt.gca()
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def plot_basis(X, cluster_labels, centroids=None, x_index=0, y_index=1,
               title="Cluster plot"):
    plt.figure()
    plt.scatter(X[:, x_index], X[:, y_index], c=cluster_labels)
    if(centroids is not None):
        plt.scatter(centroids[:, x_index], centroids[:, y_index],
                marker='x', s=169, linewidths=3, color='r', zorder=10)
    plt.title(title)
    plt.show()
    

def plot_mds(X, cluster_labels, title="MDS plot", r=1):
    mds = MDS(n_components=2, metric=False, random_state=r).fit_transform(X)
    plt.figure()
    plt.scatter(mds[:, 0], mds[:, 1], c=cluster_labels)
    plt.title(title)
    plt.show()
    

def plot_dendrogram(X, method="ward", truncate_mode=None, title="Dendrogram"):
    plt.figure()
    h_d = sch.dendrogram(sch.linkage(X, method=method),
                         truncate_mode=truncate_mode)
    plt.title(title)
    plt.show()
    
def plot_knn_distance(X, k=5, title="KDD distance"):
    nn_c = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nn_c.kneighbors(X)
    k_dis = np.sum(distances, axis=1)/(k -1)
    k_dis = np.sort(k_dis)
    plt.figure()
    plt.plot(k_dis)
    plt.ylabel("Distance")
    plt.xlabel("Number of data")
    plt.title(title)
    plt.show()


def plot_dbscan(X, dbscan):
    labels = dbscan.labels_
    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.title('DBSCAN: estimat number of clusters - %d ' % n_cluster)
    plt.show()



def plot_k_elbow(X, n_clusters):
    see = []
    for i in range(1, (n_clusters+1)):
        kmeans = KMeans(n_clusters=i, random_state=1).fit(X)
        see.append(kmeans.inertia_)
    plt.plot(np.arange(1, n_clusters+1), see, marker="o")
    plt.xlabel("K")
    plt.ylabel("SEE")
    plt.title("Choosing Optimal K with SEE")
    plt.show()

def plot_k_with_silhouette(X, n_clusters):
    sil_score = []
    for i in range(2, (n_clusters+1)):
        pred = KMeans(n_clusters=i, random_state=1).fit_predict(X)
        sil_score.append(silhouette_score(X, pred))
    plt.plot(np.arange(2, n_clusters+1), sil_score, marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("Choosing Optimal K with Silhouette Score")
    plt.show()

if __name__ == '__main__':
    raw_data = pd.read_csv("Absenteeism_at_work.csv", sep=";")
    X = raw_data.values
    k_means = KMeans(n_clusters=3)
    pred = k_means.fit_predict(X)
    plot_silhouette(X, pred)
    
