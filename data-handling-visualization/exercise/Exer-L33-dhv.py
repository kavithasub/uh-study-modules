# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:58:10 2023

@author: kthat
"""

# Excercise L33 - DHV
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x = np.linspace(0, 10, 50)
dy = 0.8  # also could be a numpy array 
y = np.sin(x) + dy * np.random.randn(50)

plt.figure(figsize=(5,4))
plt.errorbar(x, y, yerr=dy, fmt='.k');


plt.figure(figsize=(5,4))
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0);


a = np.linspace(0, 10, 50)
dy = 0.4 * np.random.randn(30)  
b = np.sin(x) + dy

plt.figure(figsize=(5,4))
plt.errorbar(a, b, yerr=dy, fmt='.k');