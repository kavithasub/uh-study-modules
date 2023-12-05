# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:38:10 2023

@author: kthat
"""
# Exercise 10 - ADS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.optimize as opt

def exponential_growth(t, scale, growth) :
    f = scale * np.exp(growth * (t - 1990))
    return f
    

data = pd.read_csv("C:\\Users\\kthat\\OneDrive\\ADS1\\india_population.csv")

df = pd.DataFrame(data)
print(df.head())


# fit exponential growth
popt, pcorr = opt.curve_fit(exponential_growth, df["date"], df["Population"])

print("fit parameter 1950: ", popt[0])
print("growth rate : ", popt[1])
df["popExponential"] = exponential_growth(df["date"], *popt)

plt.figure()
plt.plot(df["date"], exponential_growth(df["date"], popt[0], popt[1]), label = "trial fit")
plt.plot(df["date"], df["Population"])
#plt.plot(df["date"], df["popExponential"])
plt.xlabel("Year")
plt.legend()
plt.show()