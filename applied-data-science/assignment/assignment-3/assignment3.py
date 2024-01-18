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


def read_data(datafile):
    """
    Read data from excel file and return dataframe
    """
    data = pd.read_excel(datafile)
    return data


def mani_data(df, indicators):
    # Pivot the data for plotting
    df = pd.DataFrame(data)
    df_pivot = df.reset_index().groupby(['Country Name', 'Series Name'])[
        '2010 [YR2010]'].aggregate('first').unstack()
    #df_pivot = df.pivot(index=df.index, columns='Series Name')['2010 [YR2010]']
    # df_pivot = df.pivot_table(
    #   values='2010 [YR2010]', index=df.index, columns='Series Name', aggfunc='first')
    #df_pivot = df_pivot.dropna()
    return df_pivot


def exponential_growth(t, scale, growth):
    f = scale * np.exp(growth * (t - 1990))
    return f


def print_to_file(df_m):
    """
    This method is common print used to print given data to text file
    """
    x = np.random.randint(1, 100)
    filename = "dataframe" + str(x) + ".txt"
    textfile = open(filename, "w")
    df_m.to_string(textfile)
    textfile.close()


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
#df_new['2010'] = round(df_new['2010'], 3)

df_m['2010'].replace({'..': '0'}, inplace=True)
#df_m = pd.to_numeric(df_m['2010'])

# print(type(df_m['2010'][0]))
df_m = df_m.dropna()
#df_m['2010'] = df_m['2010'].astype(float)
# print(df_new.head())
df_m = df_m.reset_index().groupby(['Country Name', 'Series Name'])[
    '2010'].aggregate('first').unstack()

df_m = df_m.reset_index(drop=True)

print(df_m.head())

##corr = df_m.corr()

# df_clus = df_m[["People using at least basic drinking water services (% of population)",
#                 "Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)"]].copy()

df_clus = df_m[["Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)",
                "Renewable internal freshwater resources, total (billion cubic meters)"]].copy()


df_clus = df_clus.astype(float)
df_clus.to_excel('test3.xlsx')
df_clus2, df_clus_min, df_clus_max = ct.scaler(df_clus)


ncluster = 4

#plt.figure(figsize=(8, 8))
#cm = plt.colormaps["Paired"]
x = df_clus["Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)"]
y = df_clus["Renewable internal freshwater resources, total (billion cubic meters)"]
#plt.scatter(x, y, ncluster, marker="d", cmap=cm)
#plt.xlabel("Using drinking water population %")
#plt.ylabel("Water productivity")
#plt.show()


kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
kmeans.fit(df_clus2)

labels = kmeans.labels_
cen = kmeans.cluster_centers_
# print(cen)

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




'''

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
'''
