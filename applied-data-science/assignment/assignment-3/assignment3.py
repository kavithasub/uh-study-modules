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
import sklearn.metrics as skmet
import scipy.optimize as opt


def read_data(datafile):
    """
    Read data from excel file and return dataframe
    """
    data = pd.read_excel(datafile, skipfooter=5)
    return data


def manipulate_data(df_m):
    """
    Manipulate basic dataframe
    """
    # Remove unwanted columns and rename columns
    df_m = df_m.reset_index(drop=True)
    df_m = df_m.drop(columns=['Country Code', 'Series Code'])
    df_m = df_m.rename(columns={'1970 [YR1970]': '1970', '1980 [YR1980]': '1980',
                                '1990 [YR1990]': '1990', '2000 [YR2000]': '2000',
                                '2010 [YR2010]': '2010', '2020 [YR2020]': '2020'})
    df_m = df_m.dropna()

    cols = df_m.columns[2:]  # exclude first two columns
    # Replace null values with 0
    for i, r in enumerate(cols):
        df_m[cols[i]].replace({'..': '0'}, inplace=True)
        i = i+1

    # Make values to float64 type
    df_m[cols] = df_m[cols].apply(pd.to_numeric)

    # Transpose dataframe
    df_m_transpose = df_m.transpose()

    return df_m, df_m_transpose


def prepare_data_for_cluster(df_m, year, indicators):
    """
    Using manipulated dataframe prepare data for cluster
    """
    df_m = df_m.reset_index().groupby(['Country Name', 'Series Name'])[
        year].aggregate('first').unstack()
    df_m = df_m.reset_index(drop=True)

    # Rename appropriate indicators to explore in scatter matrix or draw heatmap
    df_m = df_m.rename(columns={indicators[0]: 'F.W.W.AGRI',  # fresh water withdrwal for agriculture
                                # total fresh water withdrwal
                                indicators[1]: 'F.W.W.TOTAL',
                                # fresh water withdrwal for industry
                                indicators[2]: 'F.W.W.INDUS',
                                # Agriculture GDP value added
                                indicators[3]: 'AGRI.GDP',
                                # renewable internal fresh water
                                indicators[4]: 'R.INT.F.W'
                                })
    #                                # drinking water population
                                   # indicators[3]: 'D.W.POPUL',
    df_explore = df_m[['F.W.W.AGRI', 'F.W.W.TOTAL', 'F.W.W.INDUS',
                                  'AGRI.GDP', 'R.INT.F.W']].copy()
    return df_explore


def exponential_growth(t, scale, growth):
    f = scale * np.exp(growth * (t - 1990))
    return f


def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """

    f = a / (1.0 + np.exp(-k * (t - t0)))

    return f


''' 
Step1 : Read data file into dataframe and manipulate the dataframe
'''
dataframe = read_data("Water_withdrawal_agriculture_limited_indicators.xlsx")

df_m = dataframe.copy()
df_m, df_m_transpose = manipulate_data(df_m)

# df_m.to_excel("test1.xlsx")
# df_m_transpose.to_excel("test2.xlsx")

'''
Step 2 : Prepare data for cluster
'''

year = '2020'
indicators = ['Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)',
              'Annual freshwater withdrawals, total (billion cubic meters)',
              'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
              'Agriculture, forestry, and fishing, value added (% of GDP)',
              'Renewable internal freshwater resources, total (billion cubic meters)']

#'People using at least basic drinking water services (% of population)',
df_explore = prepare_data_for_cluster(df_m, year, indicators)

# Checking correlations as highly correlated categories are not good for clustering
corr = df_explore.corr(numeric_only=True)

# Get heatmap
#plt.figure(figsize=[8, 8])
#plt.imshow(corr)
#plt.colorbar()

# Explore data set as scatter matrix
pd.plotting.scatter_matrix(df_explore, figsize=(10, 10), s=10)
plt.show()

# By looking at scatter matrix selected two indicators for further proccesing
# F.W.W.AGRI => Annual freshwater withdrawals, agriculture
# AGRI.GDP => Agriculture, forestry, and fishing GDP
# Exract two indicators for climate change fitting
clus_col_1 = 'F.W.W.AGRI'
clus_col_2 = 'AGRI.GDP'
df_clus = df_explore[[clus_col_1, clus_col_2]].copy()
#df_clus = df_clus.astype(float)

'''
Step 3 : Clustering - Normalise data, find ncluster and plot
'''
# Normalise dataframe
df_clus, df_clus_min, df_clus_max = ct.scaler(df_clus)
print(df_clus.describe())


# Find cluster number using silhoutte score 
# loop over trial numbers of clusters calculating the silhouette
for i in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=i, n_init=20)
    kmeans.fit(df_clus)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    score = skmet.silhouette_score(df_clus, labels)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")



# Plot for 4 clusters

#plot_cluster()
ncluster = 3

x = df_clus[clus_col_1]
y = df_clus[clus_col_2]

#set up kmeans and fit
kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
kmeans.fit(df_clus)

labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(8.0, 8.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(x, y, c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# show cluster centres
xc = cen[:, 0]
yc = cen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=50)
# c = colour, s = size

plt.xlabel("Fresh water withdraw for Agriculture (%)")
plt.ylabel("Renewable freshwater resources (milli)")
plt.title("4 clusters")
plt.show()


'''

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


'''
