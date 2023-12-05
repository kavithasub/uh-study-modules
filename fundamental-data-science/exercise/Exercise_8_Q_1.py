# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:21:05 2023

@author: kthat
"""
# Fundamental - Exercise 8 - Ques 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\kthat\\OneDrive\\FDS\\electrons.csv", delimiter=' ')
print(data)
x = np.array(data['Distance'])
y = np.array(data['Energy'])

plt.figure(dpi=144)
plt.scatter(x, y)

answer = np.polyfit(x, y, 1)
print(answer)
a = answer[1]
b = answer[0]
yfit = a + b*x

plt.plot(x, yfit, color='r')

efield = b / 1.6e-19


