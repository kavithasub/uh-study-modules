# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:40:01 2023

@author: kthat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import seaborn as sns
import stats as stats


def read_data(datafile):
    """
    Read data from excel file and return dataframe
    """
    data = pd.read_excel(datafile)
    return data


def print_to_file(df_m):
    """
    This method is common print used to print data after manipulated dataframe
    """
    x = np.random.randint(1, 100)
    filename = "dataframe" + str(x) + ".txt"
    textfile = open(filename, "w")
    df_m.to_string(textfile)
    textfile.close()


def manipulate_data(df, ind_header, ind_value, years):
    """
    Filter data, manipulate, and transpose dataframe to draw plot from it
    """
    # Groupby indicator values
    df_m = df.groupby(ind_header, group_keys=True)
    df_m = df_m.get_group(ind_value)
    df_m = df_m.reset_index()
    df_m = df_m.set_index('Country Name')
    df_m = df_m.loc[:, years]

    # Remove null values
    df_m = df_m.dropna()

    # Transpose dataframe
    df_transpose = df_m.transpose()
    return df_transpose, df_m


def draw_barplot(df, title):
    """ 
    This method is used to create a bar plot.
    Arguments: x and y will be selectively assigned by method attributes. 
    """

    fig, ax = plt.subplots()
    ax = df.plot.bar()

    # Add label and title
    ax.set_title(title, color="red", fontsize=10)

    ax.legend(loc="upper right", fontsize='x-small')
    plt.show()
    return


def draw_heatmap(corr) :
    """ 
    This method is used to create a heatmap.
    Arguments: x and y will be selectively assigned by method attributes. 
    """
    plt.figure(figsize=(40, 20))
    sns.heatmap(corr, annot=True, annot_kws={'size':32})
    #sns.heatmap(correlation_matrix, annot=True, cmap='mako', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of country - Ireland')
    plt.savefig('heatmap.png')

    plt.show()
    return


def prepare_stats(df, indicators, years, coun_header, coun_value) :
    """ 
    This method is used to prepare dataset to retrieve statistics data and find correlation matrix.
    """
    # Groupby indicator values
    df_m = df.groupby(coun_header, group_keys=True)
    df_m = df_m.get_group(coun_value)
    df_m = df_m.reset_index()
    df_m = df_m.set_index('Series Name')
    df_m = df_m.loc[:, years]

    # Remove null values
    df_m = df_m.dropna()

    # Transpose dataframe
    df_transpose = df_m.transpose()
    correlation = df_transpose.corr()
    return df_m, df_transpose, correlation
    


filename = 'C:\\Users\\kthat\\OneDrive\\ADS1\\assignment2\\World_Development_Indicators_new2.xlsx'
dataframe = read_data(filename)


"""
Prepare data with manipulating dataframe and ploting bar charts for two dataframes
"""
indicator_header = 'Series Code'
indicator_value_1 = 'EN.CO2.BLDG.ZS' # indicator selected is - CO2 emissions from residential buildings and commercial and public services (% of total fuel combustion)
indicator_value_2 = 'SP.URB.GROW' # indicator selected is - Urban population growth (annual %)
years = ['1992 [YR1992]', '1996 [YR1996]', '2000 [YR2000]', '2004 [YR2004]', '2008 [YR2008]', '2012 [YR2012]']
# Call method for manipulate dataframe
df_transposed_1, df_climate_1 = manipulate_data(
    dataframe, indicator_header, indicator_value_1, years)
df_transposed_2, df_climate_2 = manipulate_data(
    dataframe, indicator_header, indicator_value_2, years)

# Print transposed data
#print_to_file(df_transposed_1)
#print_to_file(df_transposed_2)

# Call method to plot bar chart
title_bar_1 = "CO2 emissions from residential buildings and commercial and public services"
title_bar_2 = " Urban population growth (annual %)"
#draw_barplot(df_climate_1, title_bar_1)
#draw_barplot(df_climate_2, title_bar_2)

"""
Prepare data for heatmap and ploting heatmap for Ireland.

Argument 'indicators' created using 'Series Code' respective of below 'Series Name''
EN.CO2.MANF.ZS = CO2 emissions from manufacturing industries and construction (% of total fuel combustion)
EN.CO2.BLDG.ZS = CO2 emissions from residential buildings and commercial and public services (% of total fuel combustion)
NV.AGR.TOTL.KD.ZG = Agriculture, forestry, and fishing, value added (annual % growth)
NV.IND.TOTL.KD.ZG = Industry (including construction), value added (annual % growth)
SP.URB.GROW = Urban population growth (annual %)
SP.RUR.TOTL.ZG = Rural population growth (annual %)
EN.CO2.OTHX.ZS = CO2 emissions from other sectors, excluding residential buildings and commercial and public services (% of total fuel combustion)
EN.ATM.METH.AG.KT.CE = Agricultural methane emissions (thousand metric tons of CO2 equivalent)
"""
indicators = ['EN.CO2.MANF.ZS', 'EN.CO2.BLDG.ZS', 'NV.AGR.TOTL.KD.ZG', 'NV.IND.TOTL.KD.ZG', 'SP.URB.GROW', 'SP.RUR.TOTL.ZG', 'EN.CO2.OTHX.ZS']
years = [ '1996 [YR1996]', '2000 [YR2000]', '2004 [YR2004]', '2008 [YR2008]', '2012 [YR2012]']
country_header = 'Country Name'
country = 'Ireland'
df_manipulate, df_heatmap, correlation = prepare_stats(dataframe, indicators, years, country_header, country)
# print heatmap data
print_to_file(df_heatmap)

# Call method to plot heatmap
draw_heatmap(correlation)

"""
Summary of Statistcis data
"""
# Find describe stats variables
describe_stats = df_manipulate.describe()
print_to_file(describe_stats)
# Find skewness of distribution
skewness = stats.skew(df_manipulate)
print_to_file(skewness)
print(skewness)






