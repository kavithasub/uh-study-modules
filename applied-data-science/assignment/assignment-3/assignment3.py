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


def print_to_file(df_m):
    """
    This method is common print used to print given data to text file
    """
    x = np.random.randint(1, 100)
    filename = "dataframe" + str(x) + ".txt"
    textfile = open(filename, "w")
    df_m.to_string(textfile)
    textfile.close()


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
    df_explore = df_2[['F.W.W.AGRI', 'F.W.W.TOTAL', 'F.W.W.INDUS',
                       'AGRI.GDP', 'R.INT.F.W']].copy()

    return df_explore, df_country


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


def create_cluster(x, y, labels):  # , df_clus , predict):
    plt.figure(figsize=(8.0, 8.0))
    #fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=140)
    # scatter plot with colours selected using the cluster numbers
    # axs[0].scatter(p1_norm[y_predict1 == 0, 0], p1_norm[y_predict1 == 0, 1], s=50,c='lightcoral', label='cluster 0')
    #ax.scatter(df_clus['Cluster' == 0, 0], df_clus['Cluster' == 0, 1], c='blue', cmap="tab10", label='cluster 0')
    #ax.scatter(df_clus['Cluster' == 1, 0], df_clus['Cluster' == 1, 1], c='blue', cmap="tab10", label='cluster 1')
    #ax.scatter(df_clus['Cluster' == 2, 0], df_clus['Cluster' == 2, 1], c='blue', cmap="tab10", label='cluster 2')
    plt.scatter(x, y, c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours

    # show cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=50)
    # c = colour, s = size

    plt.xlabel("Fresh water withdraw Industry")
    plt.ylabel("Agri GDP ")
    plt.title("clusters")
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
    df_fit_ = df_fit_.drop(columns=['Country Code', 'Series Code'])

    cols = df_fit_.columns[2:]  # exclude first two columns
    # Replace null values with 0
    for i, r in enumerate(cols):
        df_fit_[cols[i]].replace({'..': '0'}, inplace=True)
        i = i+1
        
    # Make values to float64 type
    df_fit_[cols] = df_fit_[cols].apply(pd.to_numeric)

    # Select required indicator and by country
    df_fit_ = df_fit_[df_fit_['Series Name'] == indicator]
    df_fit_ = df_fit_[df_fit['Country Name'] == country]
    
    df_fit_ = df_fit_.reset_index(drop=True)
    df_fit_ = df_fit_.iloc[:, 2:]
    df_fit_ = df_fit_.transpose()
    df_fit_.columns = ['AGRI_GDP']
    df_fit_ = df_fit_.reset_index().rename(columns={'index':'Year'})
    
    return df_fit_



def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """    
    f = a / (1.0 + np.exp(-k * (t - t0)))
    
    return f


def exponential_growth(t, scale, growth):
    f = scale * np.exp(growth * (t - 1970))
    return f


def plot_exponential_growth(df_fit_country, country):
    popt, pcorr = opt.curve_fit(exponential_growth, df_fit_country["Year"], df_fit_country["AGRI_GDP"],
                                p0=[4e8, 0.03])
    print("Fit parameter with exponential function", popt)
    
    df_fit_country["pop_exp"] = exponential_growth(df_fit_country["Year"], *popt)
    plt.figure()
    plt.plot(df_fit_country["Year"], df_fit_country["AGRI_GDP"], label="data")
    plt.plot(df_fit_country["Year"], df_fit_country["pop_exp"], label="fit")

    plt.legend()
    plt.title(f"Fit exponential growth for {country}")
    plt.show()

    print("Agriculture Value added as % of GDP")
    print("2030:", exponential_growth(2030, *popt) / 1.0e6, "Mill.")
    print("2040:", exponential_growth(2040, *popt) / 1.0e6, "Mill.")
    print("2050:", exponential_growth(2050, *popt) / 1.0e6, "Mill.")
    
    return


def plot_logistic(df_fit_country, country):
    popt, pcovar = opt.curve_fit(logistics, df_fit_country["Year"], df_fit_country["AGRI_GDP"], 
                                p0=(16e8, 0.04, 1990.0))
    print("Fit parameter with logistic function", popt)

    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.linspace(1980, 2040)
    pop_logistics = logistics(years, *popt)

    sigma = error.error_prop(years, logistics, popt, pcovar)
    low = pop_logistics - sigma
    up = pop_logistics + sigma


    plt.figure()
    plt.title(f"Agriculture Value added GDP Forcast for {country}")
    plt.plot(df_fit_country["Year"], df_fit_country["AGRI_GDP"], label="data")
    plt.plot(years, pop_logistics, label="fit")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.5, color="y")

    plt.legend(loc="upper left")
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

# df_m.to_excel("df_m.xlsx")
# df_m_transpose.to_excel("df_m_trans.xlsx")

'''
Step 2 : Prepare data for cluster
--------------------------------------
'''

year = '2020'
indicators = ['Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)',
              'Annual freshwater withdrawals, total (billion cubic meters)',
              'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)',
              'Agriculture, forestry, and fishing, value added (% of GDP)',
              'Renewable internal freshwater resources, total (billion cubic meters)']

# Copy manipulated data df_m to explore, so df_m will be reuse
df_explore = df_m.copy()

df_explore, df_country = prepare_data_for_cluster(df_explore, year, indicators)
# df_country.to_excel('df_country.xlsx')
# Checking correlations as highly correlated categories are not good for clustering
corr = df_explore.corr(numeric_only=True)

# Get heatmap
#plt.figure(figsize=[8, 8])
# plt.imshow(corr)
# plt.colorbar()

# Explore data set as scatter matrix
pd.plotting.scatter_matrix(df_explore, figsize=(10, 10), s=10)
plt.show()

# By looking at scatter matrix selected two indicators for further proccesing
# F.W.W.AGRI => Annual freshwater withdrawals, agriculture
# AGRI.GDP => Agriculture, forestry, and fishing GDP
# Exract two indicators for climate change fitting
clus_col_1 = 'F.W.W.INDUS'
clus_col_2 = 'AGRI.GDP'
df_clus = df_explore[[clus_col_1, clus_col_2]].copy()
#df_clus = df_clus.astype(float)

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
# ncluster selected based on silouette score
ncluster = 3
x = df_clus[clus_col_1]
y = df_clus[clus_col_2]

# set up kmeans and fit
kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
kmeans.fit(df_clus)

# Set lables to each data point
labels = kmeans.labels_
# Get estimated cluster centers
cen = kmeans.cluster_centers_

# Call cluster method to plot
create_cluster(x, y, labels)

# Exract one country from each cluster
df_country['Cluster No'] = labels
view_cluster_countries(df_country, ncluster)
# By looking at printed countries list, selected one country from each cluster


'''
Step 4 : Fitting model
--------------------------
'''
# According to cluster clasification, selected below countries
# Cluster 0 => Ghana; Cluster 1 => France; Cluster 2 => Cyprus
# Prepare data for fitting

countries = ['Ghana', 'France', 'Cyprus']
indicator = 'Agriculture, forestry, and fishing, value added (% of GDP)'

# Read data file and create dataframe
gdp_data = read_data('Agricuture_GDP_countries.xlsx')
df_fit = gdp_data.copy()

# Prepare data for fitting
df_fit_Ghana = prepare_data_for_fit(df_fit, countries[0], indicator)
df_fit_France = prepare_data_for_fit(df_fit, countries[1], indicator)
df_fit_Cyprus = prepare_data_for_fit(df_fit, countries[2], indicator)


# fit exponential growth
# Plot data to exponential growth
#plot_exponential_growth(df_fit_Ghana, countries[0])
#plot_exponential_growth(df_fit_France, countries[1])
#plot_exponential_growth(df_fit_Cyprus, countries[2])


# fit logistic
# Plot data to logistic function
#plot_logistic(df_fit_Ghana, countries[0])
plot_logistic(df_fit_France, countries[1])
#plot_logistic(df_fit_Cyprus, countries[2])


'''
End of file
'''
