# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:03:41 2023

@author: kthat
"""

# coding project FDS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data2-1.csv')
# hist contains numbers of entries in each bin, edge contains bin boundaries

hist, edge = np.histogram(data, bins=32, range=[0.0, 160000.0]) #number of equal-width bins in the given range
print('hist =>',hist)
print('edge =>', edge)

# calculate bin centre locations and bin widths
xdist=0.5*(edge[1:]+edge[:-1])
wdist=edge[1:]-edge[:-1]

# normalise the distribution
#ydist is a discrete PDF

ydist = hist/np.sum(hist)
print('x distribution => ', xdist, 'y distribution => ', ydist)

#cumulative distribution
cdst=np.cumsum(ydist)

# plot the probability distribution function
plt.figure(0)
plt.bar(xdist, ydist, width=0.9*wdist)

plt.xlabel('Salary, £', fontsize=15)
plt.ylabel('Probability', fontsize=15)

# calculate mean salary
xmean=np.sum(xdist*ydist)
print('mean => ', xmean)
# plot mean salary
plt.plot([xmean,xmean],[0.0,max(ydist)], c='red')
text = ''' Mean= {}'''.format(xmean.astype(int))
plt.text(x=xmean, y=max(ydist), s=text, fontsize=10, c='red')


#find percentage of people have salary above X
hdist=ydist*(xdist >= 80000.0)
print(xdist >= 80000.0)
#(xdist >= 100000.0) = hdist / ydist
print(np.sum(hdist)) #= 5/100.0
print('h dist => ', hdist)


plt.bar(xdist, hdist, width=0.9*wdist, color='orange')
hsum=np.sum(hdist)*100.0
text = ''' {}% of salary above £80k'''.format(hsum.astype(int))
plt.text(x=80000, y=0.03, s=text, fontsize=10, c='orange')


