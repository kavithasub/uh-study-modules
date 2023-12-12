# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:40:01 2023

@author: kthat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts


def read_data(datafile):
    """
    Read data from excel file and return dataframe
    """
    data = pd.read_excel(datafile)
    #df = pd.DataFrame(data)
    return data


def print_dataframe(df_m):
    """
    This method is common print used to print data after manipulated dataframe
    """
    x = np.random.randint(1, 100)
    filename = "dataframe" + str(x) + ".txt"
    textfile = open(filename, "w")
    df_m.to_string(textfile)
    textfile.close()


def manipulate_data(dataframe, name, value):
    """
    Filter data, manipulate, and transpose dataframe to draw plot from it
    """
    df_m = dataframe.groupby(name, group_keys=True)
    df_m = df_m.get_group(value)
    #df_m = df_m.reset_index()
    df_m = df_m.set_index('Country Name')

    # Remove null values
    df_m = df_m.dropna()

    # Transpose dataframe
    #df_transpose = df_m.reset_index()
    #df_transpose = df_transpose.set_index('Country Name')
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

    ax.legend()
    plt.show()

    return


filename = 'C:\\Users\\kthat\\OneDrive\\ADS1\\assignment2\\World_Development_Indicators_new.xlsx'
dataframe = read_data(filename)
# print_dataframe(dataframe)


"""
Manipulate data and ploting bar charts for two different dataframes
"""
indicator_header = 'Series Code'
# indicator selected is - CO2 emissions from manufacturing industries and construction (% of total fuel combustion)
indicator_value_1 = 'NV.IND.TOTL.KD.ZG'
indicator_value_2 = 'SP.URB.GROW'
indicator_value_3 = 'EN.CO2.BLDG.ZS'
# Call method for manipulate dataframe
df_transposed_1, df_climate_1 = manipulate_data(
    dataframe, indicator_header, indicator_value_1)
df_transposed_2, df_climate_2 = manipulate_data(
    dataframe, indicator_header, indicator_value_2)
#df_transposed_3, df_climate_3 = manipulate_data(dataframe, indicator_header, indicator_value_3)

# Print transposed data
# print_dataframe(df_transposed_1)
# print_dataframe(df_transposed_2)

# Call method to plot bar chart
title_bar_1 = "industry anual growth"
title_bar_2 = "urban pop"
title_bar_3 = "co2 emmis from resident"
draw_barplot(df_climate_1, title_bar_1)
draw_barplot(df_climate_2, title_bar_2)
#draw_barplot(df_climate_3, title_bar_3)
