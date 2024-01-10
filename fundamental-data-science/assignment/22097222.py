# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:03:41 2023

@author: kthat
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_mean(xdist, ydist):
    """This method is to calculate mean salary value"""
    # calculate mean salary
    xmean = np.sum(xdist*ydist)
    print(xmean)
    # plot mean salary
    plt.plot([xmean, xmean], [0.0, max(ydist)], label='Mean Annual Salary', c='red', linewidth=2)
    text = ''' W= {}'''.format(xmean.astype(int))
    plt.text(x=xmean, y=max(ydist), s=text, fontsize=10, c='red')
    plt.legend()

    return xmean


def find_x(edge, xdist, wdist, ydist, xmean):
    """This method is to calculate X value"""
    # cumulative distribution
    cdst = np.cumsum(ydist)
    # find value X such that 0.05 of the distribution corresponds to values >X value
    indx = np.argmin(np.abs(cdst-(1.0-0.05)))
    xhigh = edge[indx]
    print(xhigh)
    plt.bar(xdist[indx:], ydist[indx:], width=0.9*wdist[0:indx], color='orange')
    plt.plot([xhigh, xhigh], [0.0, max(ydist)], c='green', linestyle='dashed', label='X value')
    text = ''' 5% of people's salary above £{}'''.format(xhigh.astype(int))
    plt.text(x=80000, y=0.03, s=text, fontsize=9, c='green')
    plt.legend()
    plt.savefig('histo.png')

    return

# Read data file
data = pd.read_csv('data2-1.csv')

# Create probability density function
# 32 number of equal-width bins in the range 0-160000
hist, edge = np.histogram(data, bins=32, range=[0.0, 160000.0])

# calculate bin centre locations and bin widths
xdist = 0.5*(edge[1:]+edge[:-1]) # xdist is bin centre location
wdist = edge[1:]-edge[:-1] # wdist is bin width

# normalise the distribution
ydist = hist/np.sum(hist)  # ydist is a discrete PDF

# plot the PDF
plt.figure(0)
plt.bar(xdist, ydist, width=0.9*wdist)
plt.plot(xdist, ydist, label='PDF', color = 'purple')
plt.xlabel('Salary, £', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Probability Density Function', fontsize=15, color='blue')
plt.legend()


xmean = find_mean(xdist, ydist)

find_x(edge, xdist, wdist, ydist, xmean)
