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


def manipulate_data(df, ind_header, ind_value, years, countries):
    """
    Filter data, manipulate, and transpose dataframe to draw plot from it
    """
    # Groupby indicator values
    df_m = df.groupby(ind_header, group_keys=True)
    df_m = df_m.get_group(ind_value)
    df_m = df_m.reset_index()
    df_m = df_m.set_index('Country Name')
    df_m = df_m.loc[:, years]
    df_m = df_m.loc[countries, :]

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


def draw_heatmap(corr, country, cmap):
    """ 
    This method is used to create a heatmap.
    Arguments: x and y will be selectively assigned by method attributes. 
    """
    plt.figure(figsize=(40, 20))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, annot_kws={'size': 32})
    plt.title('Correlation Heatmap of country ' + country, fontsize=10)

    plt.show()
    return


def prepare_stats(df, years, coun_header, coun_value):
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


def draw_lineplot(df_m):
    """ 
    This method is used to create a line plot.
    Arguments: x and y will be selectively assigned by method attributes. 
    """
    plt.figure()
    df_m.plot.line()
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()

    return


def draw_table(df_m) :
    """ 
    This method is used to create a fine table.
    """
    # Create a table using matplotlib
    plt.figure(figsize=(5, 3))
    # Turn off axis labels and ticks for good appearance
    #plt.axis('off')
    plt.title('Urban population by country and year')
    # Display the table
    table = plt.table(cellText=df_m1.values, rowLabels=df_m1.index,
                      colLabels=df_m1.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(df_m1.columns))))

    return


# Step1: Read world bank data from file
filename = 'C:\\Users\\kthat\\OneDrive\\ADS1\\assignment2\\World_Development_Indicators_new3.xlsx'
dataframe = read_data(filename)


"""
Prepare data with manipulating dataframe and ploting bar charts for two dataframes
"""
indicator_header = 'Series Code'
# indicator selected is - CO2 emissions from other sectors, excluding residential buildings and commercial and public services (% of total fuel combustion)
indicator_value_1 = 'EN.CO2.OTHX.ZS'
# indicator selected is - Industry (including construction), value added (annual % growth)
indicator_value_2 = 'NV.IND.TOTL.KD.ZG'
years = ['1996 [YR1996]', '2000 [YR2000]',
         '2004 [YR2004]', '2008 [YR2008]', '2012 [YR2012]']
countries = ['Norway', 'Australia', 'Ireland', 'Netherlands', 'Denmark', 'Iceland', 'Canada', 'United States']
# Call method for manipulate dataframe
df_transposed_1, df_climate_1 = manipulate_data(
    dataframe, indicator_header, indicator_value_1, years, countries)
df_transposed_2, df_climate_2 = manipulate_data(
    dataframe, indicator_header, indicator_value_2, years, countries)

# Print transposed data
print_to_file(df_transposed_1)
print_to_file(df_transposed_2)

# Call method to plot bar chart
title_bar_1 = "CO2 emissions from all other sectors, excluding residential buildings and commercial or public services"
title_bar_2 = "Industry (including construction), value added (annual % growth)"
draw_barplot(df_climate_1, title_bar_1)
draw_barplot(df_climate_2, title_bar_2)


"""
Prepare data for heatmap and ploting heatmap for selected countries.
"""
years = ['1996 [YR1996]', '2000 [YR2000]',
         '2004 [YR2004]', '2008 [YR2008]', '2012 [YR2012]']
country_header = 'Country Name'

# Create heatmap for 'Ireland'
country = 'Ireland'
cmap = 'mako'
df_manipulate, df_heatmap, correlation = prepare_stats(
    dataframe, years, country_header, country)
# print heatmap data
print_to_file(df_heatmap)
# Call method to plot heatmap
draw_heatmap(correlation, country, cmap)


"""
Summary of Statistcis data
"""
# Find describe stats variables
describe_stats = df_manipulate.describe()
print_to_file(describe_stats)
# Find skewness of distribution and print
skewness = stats.skew(df_manipulate)
print(skewness)
# Find kurtosis of distribution print
kurtosis = stats.kurtosis(df_manipulate)
print(kurtosis)


"""
Create another two heatmap for different countries
"""
# Create heatmap for 'United States'
country = 'United States'
cmap = 'viridis'
df_manipulate, df_heatmap, correlation = prepare_stats(
    dataframe, years, country_header, country)
print_to_file(df_heatmap)
draw_heatmap(correlation, country, cmap)

# Create heatmap for 'Netherlands'
country = 'Netherlands'
cmap = 'magma'
df_manipulate, df_heatmap, correlation = prepare_stats(
    dataframe, years, country_header, country)
print_to_file(df_heatmap)
draw_heatmap(correlation, country, cmap)


"""
Create line plots for two different indicators 
"""
indicator_header = 'Series Code'
# indicator selected is - CO2 emissions from residential buildings and commercial and public services (% of total fuel combustion)
indicator_value_1 = 'EN.CO2.BLDG.ZS'
# indicator selected is - Urban population growth (annual %)
indicator_value_2 = 'SP.URB.GROW'
years = ['1996 [YR1996]', '2000 [YR2000]',
         '2004 [YR2004]', '2008 [YR2008]', '2012 [YR2012]']
countries = ['Norway', 'Australia', 'Ireland', 'Netherlands', 'Iceland', 'Canada', 'United States']
# Call method for manipulate dataframe
df_transposed_1, df_m1 = manipulate_data(
    dataframe, indicator_header, indicator_value_1, years, countries)
df_transposed_2, df_m2 = manipulate_data(
    dataframe, indicator_header, indicator_value_2, years, countries)
print_to_file(df_transposed_1)
print_to_file(df_transposed_2)
draw_lineplot(df_transposed_1)
draw_lineplot(df_transposed_2)

"""
Create table for Urban population growth
"""
draw_table(df_m1)



