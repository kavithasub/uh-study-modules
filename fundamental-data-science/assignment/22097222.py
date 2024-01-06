# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:03:41 2023

@author: kthat
"""

# coding project FDS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_mean(xdist, ydist):
    # calculate mean salary
    xmean = np.sum(xdist*ydist)
    # plot mean salary
    plt.plot([xmean, xmean], [0.0, max(ydist)], c='red')
    text = ''' W˜= {}(mean annual salary)'''.format(xmean.astype(int))
    plt.text(x=xmean, y=max(ydist), s=text, fontsize=10, c='red')

    return xmean


def find_x(edge, xdist, wdist, ydist, xmean):
    # cumulative distribution
    cdst = np.cumsum(ydist)
    # find value X such that 0.05 of the distribution corresponds to values >X value
    indx = np.argmin(np.abs(cdst-(1.0-0.05)))
    xhigh = edge[indx]
    plt.bar(xdist[indx:], ydist[indx:], width=0.9*wdist[0:indx], color='green')
    plt.plot([xhigh, xhigh], [0.0, max(ydist)], c='orange', linestyle='dashed')
    text = ''' 5% of people's salary above £{}'''.format(xhigh.astype(int))
    plt.text(x=80000, y=0.03, s=text, fontsize=9, c='green')

    return


data = pd.read_csv('data2-1.csv')

# Create probability density function
# hist contains numbers of entries in each bin, edge contains bin boundaries
# number of equal-width bins in the given range
hist, edge = np.histogram(data, bins=32, range=[0.0, 160000.0])
print('hist =>', hist)
print('edge =>', edge)

# calculate bin centre locations and bin widths
xdist = 0.5*(edge[1:]+edge[:-1])
wdist = edge[1:]-edge[:-1]

# normalise the distribution
ydist = hist/np.sum(hist)  # ydist is a discrete PDF

# plot the PDF
plt.figure(0)
plt.bar(xdist, ydist, width=0.9*wdist)
plt.xlabel('Salary, £', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Probability Density Function', fontsize=15, color='blue')

xmean = find_mean(xdist, ydist)

find_x(edge, xdist, wdist, ydist, xmean)
