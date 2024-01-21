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
import errors as error


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


def prepare_data_for_cluster(df_2, year, indicators):
    """
    Using manipulated dataframe prepare data for cluster
    """
    df_2 = df_2.reset_index().groupby(['Country Name', 'Series Name'])[
        year].aggregate('first').unstack()
    # Keep copy of sorted dataframe for later use
    df_country = df_2.copy()
    df_country = df_country.reset_index()
    # drop index column
    df_2 = df_2.reset_index(drop=True)

    # Rename appropriate indicators to explore in scatter matrix or draw heatmap
    df_2 = df_2.rename(columns={indicators[0]: 'F.W.W.AGRI',  # fresh water withdrwal for agriculture
                                # fresh water withdrwal for industry
                                indicators[1]: 'F.W.W.INDUS',
                                # Agriculture GDP value added
                                indicators[2]: 'AGRI.GDP',
                                # renewable internal fresh water total
                                indicators[3]: 'R.INT.F.W.T',
                                # renewable internal fresh water per capita
                                indicators[4]: 'R.INT.F.W.C'
                                })

    df_explore = df_2[['F.W.W.AGRI', 'F.W.W.INDUS',
                       'AGRI.GDP', 'R.INT.F.W.T', 'R.INT.F.W.C']].copy()

    return df_explore, df_country


def plot_heatmap_scatter(df_explore, corr):
    # Get heatmap
    plt.figure(figsize=[8, 8])
    plt.imshow(corr)
    plt.colorbar()
    annotations = df_explore.columns[:]  # extract relevant headers
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=annotations)
    plt.yticks(ticks=[0, 1, 2, 3, 4], labels=annotations)

    # Explore data set as scatter matrix
    pd.plotting.scatter_matrix(df_explore, figsize=(10, 10), s=10)
    plt.show()
    return


def find_sillhoutte_score(df_clus):
    # loop over trial numbers of clusters calculating the silhouette
    for i in range(2, 7):
        # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=i, n_init=20)
        kmeans.fit(df_clus)

        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        score = skmet.silhouette_score(df_clus, labels)
        print(f"The silhouette score for {i: 3d} is {score: 7.4f}")

    return


def create_cluster(x, y, labels, cen, title):
    plt.figure(figsize=(6.0, 6.0), dpi=150)
    # scatter plot with colours selected using the cluster numbers
    plt.scatter(x, y, c=labels, cmap="tab10")
    # show cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=60)

    plt.xlabel("Fresh water withdrawal - Industry")
    plt.ylabel("Renewable freshwater resources")
    plt.title(title,
              color="red", fontsize=12)
    plt.show()

    return


def view_cluster_countries(df_country, n_clusters):
    # Print the countries in each cluster
    for n in range(kmeans.n_clusters):
        cluster_countries = df_country[df_country['Cluster No']
                                       == n]['Country Name'].tolist()
        print(f"\nCluster {n + 1} Countries:")
        print(", ".join(cluster_countries))

    return


def prepare_data_for_fit(df_fit_, country, indicator):
    # , '2021', '2022'])
    df_fit_ = df_fit_.drop(columns=['Country Code', 'Series Code'])

    cols = df_fit_.columns[2:]  # exclude first two columns
    # Replace null values with 0
    for i, r in enumerate(cols):
        df_fit_[cols[i]].replace({'..': '0'}, inplace=True)
        i = i+1

    df_fit_ = df_fit_.dropna()
    # Make values to float64 type
    df_fit_[cols] = df_fit_[cols].apply(pd.to_numeric)

    # Select required indicator and by country
    df_fit_ = df_fit_[df_fit_['Series Name'] == indicator]
    df_fit_ = df_fit_[df_fit['Country Name'] == country]

    df_fit_ = df_fit_.reset_index(drop=True)
    df_fit_ = df_fit_.iloc[:, 2:]
    df_fit_ = df_fit_.transpose()
    df_fit_.columns = ['Growth']
    df_fit_ = df_fit_.reset_index().rename(columns={'index': 'Year'})

    return df_fit_


def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))

    return f


def exponential_growth(t, scale, growth):
    f = scale * np.exp(growth * (t - 1970))
    return f


def plot_exponential_growth(df_fit_country, country, title):
    popt, pcorr = opt.curve_fit(exponential_growth, df_fit_country["Year"], df_fit_country["Growth"],
                                p0=[4e8, 0.03])
    print("Fit parameter with exponential function", popt)

    df_fit_country["growth_exp"] = exponential_growth(
        df_fit_country["Year"], *popt)
    plt.figure()
    plt.plot(df_fit_country["Year"], df_fit_country["Growth"],
             label="Freshwater per capita")
    plt.plot(df_fit_country["Year"], df_fit_country["growth_exp"], label="fit")

    plt.legend()
    plt.title(f"{title} {country}", color="red", fontsize=12)
    plt.xlabel("Year")
    plt.ylabel("Renewable freshwater(cubic meter)")
    plt.show()

    print("2030:", exponential_growth(2030, *popt) / 1.0e6, "Mill.")
    print("2040:", exponential_growth(2040, *popt) / 1.0e6, "Mill.")
    print("2050:", exponential_growth(2050, *popt) / 1.0e6, "Mill.")

    return


def plot_logistic(df_fit_country, country, title):
    popt, pcovar = opt.curve_fit(logistics, df_fit_country["Year"], df_fit_country["Growth"],
                                 p0=(4e8, 0.01, 1985.0))  # 3e12, 0.1, 1990  4e8, 0.03, 1980.0
    print("Fit parameter with logistic function", popt)

    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.linspace(1970, 2040)
    pop_logistics = logistics(years, *popt)

    sigma = error.error_prop(years, logistics, popt, pcovar)
    low = pop_logistics - sigma
    up = pop_logistics + sigma

    plt.figure()
    plt.title(f"{title} {country}", color="red", fontsize=12)
    plt.xlabel("Year")
    plt.ylabel("Renewable freshwater(cubic meter)")
    plt.plot(df_fit_country["Year"],
             df_fit_country["Growth"], label="Freshwater per capita")
    plt.plot(years, pop_logistics, label="forcast")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.5, color="y")

    plt.legend(loc="upper right")
    plt.show()

    return


''' 
Step1 : Read data file into dataframe and manipulate the dataframe
-------------------------------------------------------------------
'''
dataframe = read_data("Water_withdrawal_agriculture_limited_indicators.xlsx")

# Copy original data to manipulate
df_m = dataframe.copy()
df_m, df_m_transpose = manipulate_data(df_m)


'''
Step 2 : Prepare data for cluster
--------------------------------------
'''
year = '2020'
indicators = ['Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)',
              'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
              'Agriculture, forestry, and fishing, value added (% of GDP)',
              'Renewable internal freshwater resources, total (billion cubic meters)',
              'Renewable internal freshwater resources per capita (cubic meters)']

# Copy manipulated data df_m to explore, so df_m will be reuse
df_explore = df_m.copy()

df_explore, df_country = prepare_data_for_cluster(df_explore, year, indicators)
print(df_explore.head())
# Checking correlations as highly correlated categories are not good for clustering
corr = df_explore.corr(numeric_only=True)

# Plot heatmap and scatter-matrix
plot_heatmap_scatter(df_explore, corr)

# By looking at scatter matrix selected two indicators for further proccesing
# F.W.W.AGRI => Annual freshwater withdrawals, industry
# R.INT.F.W.T => Renewable internal freshwater resources, total (billion cubic meters)
# Exract two indicators for climate change fitting
clus_col_1 = 'F.W.W.INDUS'
clus_col_2 = 'R.INT.F.W.T'
df_clus = df_explore[[clus_col_1, clus_col_2]].copy()


'''
Step 3 : Clustering - Normalise data, find number of clusters and plot
------------------------------------------------------------------------
'''
# Normalise dataframe
df_clus, df_clus_min, df_clus_max = ct.scaler(df_clus)
print(df_clus.describe())

# Find cluster number using silhoutte score
find_sillhoutte_score(df_clus)

# Plot for clusters
# ncluster selected based on the silhouette. score for  3 is  0.7207
ncluster = 3
x = df_clus[clus_col_1]
y = df_clus[clus_col_2]
title = 'Fresh water withdrawal vs total renewable resources in 2020'

# set up kmeans and fit
kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
kmeans.fit(df_clus)

# Set lables to each data point
labels = kmeans.labels_
# Get estimated cluster centers
cen = kmeans.cluster_centers_
#scen = ct.backscale(cen, df_clus_min, df_clus_max)

# Call cluster method to plot
create_cluster(x, y, labels, cen, title)

# Exract one country from each cluster
df_country['Cluster No'] = labels
view_cluster_countries(df_country, ncluster)
# By looking at printed countries list, selected one country from each cluster


'''
Step 4 : Data Fitting and Forcasting
----------------------------------------
'''
# According to cluster clasification, selected below countries
# Cluster 0 => Ghana; Cluster 1 => Singapore; Cluster 2 => China
# Prepare data for fitting

countries = ['Ghana', 'Singapore', 'China', 'Cyprus', 'Pakistan',
             'China', 'Canada', 'Lebanon', 'Singapore']
indicator = 'Renewable internal freshwater resources per capita (cubic meters)'

# Read data file and create dataframe
gdp_data = read_data('Renewable_Fresh_Water_per_capita_1.xlsx')
df_fit = gdp_data.copy()

# Prepare data for fitting
df_fit_country_1 = prepare_data_for_fit(df_fit, countries[0], indicator)
df_fit_country_2 = prepare_data_for_fit(df_fit, countries[1], indicator)
df_fit_country_3 = prepare_data_for_fit(df_fit, countries[2], indicator)


# fit exponential growth
# Plot data to exponential growth
title = 'Renewable Freshwater Per Capita Growth '
plot_exponential_growth(df_fit_country_1, countries[0], title)
plot_exponential_growth(df_fit_country_2, countries[1], title)
plot_exponential_growth(df_fit_country_3, countries[2], title)


# fit logistic
# Plot data to logistic function
title = 'Renewable Freshwater Per Capita Forcast for'
plot_logistic(df_fit_country_1, countries[0], title)
plot_logistic(df_fit_country_2, countries[6], title)
plot_logistic(df_fit_country_3, countries[2], title)
