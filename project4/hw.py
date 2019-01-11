#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:39:38 2018

@author: leishi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy .cluster.hierarchy as sch

x = np.reshape([6, 12, 15, 23, 30, 42, 46], (-1, 1))

#fig, axes = plt.subplots(2, 2, figsize=(8, 6))
#%%
l_1 = sch.linkage(x, method="single")
print(l_1)
d_1 = sch.dendrogram(l_1, labels=x, count_sort="descending")


#%%
l_2 = sch.linkage(x, method="complete")
print(l_2)
d_2 = sch.dendrogram(l_2, labels=x, count_sort="descendent")


#%%
l_3 = sch.linkage(x, method="average")
print(l_3)
d_3 = sch.dendrogram(l_3, labels=x)


#%%
l_4 = sch.linkage(x, method="centroid")
print(l_4)
d_4 = sch.dendrogram(l_4, labels=x)


#%%
l_5 = sch.linkage(x, method="ward")
print(l_5)
d_5 = sch.dendrogram(l_5, labels=x)