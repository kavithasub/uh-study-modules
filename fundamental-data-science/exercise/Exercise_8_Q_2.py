# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:58:17 2023

@author: kthat
"""
# Fundamental - Exercise 8 - Ques 2
import numpy as np
import numpy.linalg as la
#import pandas as pd
import matplotlib.pyplot as plt

#data2 = pd.read_csv('C:\\Users\\kthat\\OneDrive\\FDS\\gauss2d.csv', delimiter=',')
data = np.genfromtxt('C:\\Users\\kthat\\OneDrive\\FDS\\gauss2d.csv', delimiter=',')
print(data)
x = data[:,0]
y = data[:,1]

plt.figure(0, dpi=144)
plt.scatter(x, y)

# find variances
xmean = np.mean(x)
ymean = np.mean(y)

varx = np.sum((x-xmean)**2)/(len(x)-1)
vary = np.sum((y-ymean)**2)/(len(y)-1)

# find covariance
covxx = varx
covyy = vary
covxy = np.sum((x-xmean)*(y-ymean)/(len(x)-1))

# covariance matrix
cm = np.array([[covxx, covxy], [covxy, covyy]])

# find eignvalue and eignvactors
eignval, eignvec = la.eig(cm)
pc1val = eignval[0]

pclx = [xmean, xmean+eignvec[0, 0]*pc1val*0.5]
pcly = [ymean, ymean+eignvec[1, 0]*pc1val*0.5]
plt.plot(pclx, pcly, color='r')
