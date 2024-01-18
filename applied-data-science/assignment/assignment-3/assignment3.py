# -*- coding: utf-8 -*-
"""
Created on Tue Dec  25 11:38:10 2023

@author: kthat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.cluster as cluster
import scipy.optimize as opt


def read_data(datafile):
    """
    Read data from excel file and return dataframe
    """
    data = pd.read_excel(datafile)
    return data


def manipulate_data(df_m):
    df_m['2010'].replace({'..': '0'}, inplace=True)
    df_m = df_m.dropna()

    df_m = df_m.reset_index().groupby(['Country Name', 'Series Name'])[
        '2010'].aggregate('first').unstack()
    df_m = df_m.reset_index(drop=True)
    
    return df_m


def exponential_growth(t, scale, growth):
    f = scale * np.exp(growth * (t - 1990))
    return f


def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    
    f = a / (1.0 + np.exp(-k * (t - t0)))
    
    return f



# Read data file
data = read_data("Water_withdrawal_agriculture_100countries_1.xlsx")
# data.corr(numeric_only=True).round(3)

# Create dataframe
df = pd.DataFrame(data)
df = df.fillna(0)
df = df.drop(columns=['Country Code', 'Series Code'])
df = df.rename(columns={'1970 [YR1970]': '1970', '1980 [YR1980]': '1980',
                        '1990 [YR1990]': '1990', '2000 [YR2000]': '2000',
                        '2010 [YR2010]': '2010', '2020 [YR2020]': '2020'})

# Manipulate dataframe
df_m = df.copy()
df_m = manipulate_data(df_m)

##corr = df_m.corr()

# Exract two indicators for climate change fitting
clus_col_1 = 'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)'
clus_col_2 = 'Renewable internal freshwater resources, total (billion cubic meters)'

df_clus = df_m[[clus_col_1, clus_col_2]].copy()
df_clus = df_clus.astype(float)
# Normalise dataframe
df_clus2, df_clus_min, df_clus_max = ct.scaler(df_clus)

# Plot for 4 clusters
ncluster = 4

x = df_clus[clus_col_1]
y = df_clus[clus_col_2]

#set up kmeans and fit
kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
kmeans.fit(df_clus2)

labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(x, y, c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# rescale and show cluster centres
scen = ct.backscale(cen, df_clus_min, df_clus_max)
# show cluster centres
xc = cen[:, 0]
yc = cen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
# c = colour, s = size

plt.xlabel("Fresh water withdraw for Agriculture (%)")
plt.ylabel("Renewable freshwater resources (milli)")
plt.title("4 clusters")
plt.show()




# fit exponential growth
popt, pcorr = opt.curve_fit(exponential_growth, df_m["year"], df_m["freshwater resources"],
                            p0=[4e8, 0.03])
# much better
print("Fit parameter", popt)

df_m["res_exp"] = exponential_growth(df["year"], *popt)
plt.figure()
plt.plot(df["date"], df["Population"], label="data")
plt.plot(df["date"], df["pop_exp"], label="fit")

plt.legend()
plt.title("Final fit exponential growth")
plt.show()
print()

print("Population in")
print("2030:", exponential_growth(2030, *popt) / 1.0e6, "Mill.")
print("2040:", exponential_growth(2040, *popt) / 1.0e6, "Mill.")
print("2050:", exponential_growth(2050, *popt) / 1.0e6, "Mill.")

