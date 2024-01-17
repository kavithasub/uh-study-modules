# -*- coding: utf-8 -*-
"""
Created on Tue Dec  25 11:38:10 2023

@author: kthat
"""

import numpy as np
import pandas as pd
import cluster_tools as ct
import matplotlib.pyplot as plt
import scipy.optimize as opt


def read_data(datafile):
    """
    Read data from excel file and return dataframe
    """
    data = pd.read_excel(datafile)
    return data


def exponential_growth(t, scale, growth):
    f = scale * np.exp(growth * (t - 1990))
    return f


data = read_data("Data_Female_employement_Agriculture.xlsx")

df = pd.DataFrame(data)
print(df.head())


df, df_min, df_max = ct.scaler(df)
print(df)
print(df_min, df_max)

# define cluster
ncluster = 2

#plot clusters into graph
plt.figure(figsize=(8, 8))
#cm = plt.colormaps["Paired"]




# fit exponential growth
popt, pcorr = opt.curve_fit(exponential_growth, df["date"], df["Population"])

print("fit parameter 1990: ", popt[0])
print("growth rate : ", popt[1])
df["popExponential"] = exponential_growth(df["date"], *popt)

plt.figure()
plt.plot(df["date"], exponential_growth(
    df["date"], popt[0], popt[1]), label="trial fit")
plt.plot(df["date"], df["Population"])
plt.xlabel("Year")
plt.legend()
plt.show()
