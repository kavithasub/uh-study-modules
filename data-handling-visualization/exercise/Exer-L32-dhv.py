# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:34:26 2023

@author: kthat
"""

# Excercise L32 - DHV
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.figure(figsize=(5,4))
plt.plot(x, y, '-ok', color='black');


plt.figure(figsize=(8,6))
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);


plt.figure(figsize=(4,3))
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='red',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);


rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.figure(figsize=(4,3))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar();


from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.figure(figsize=(4,3))
plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);




plt.show()