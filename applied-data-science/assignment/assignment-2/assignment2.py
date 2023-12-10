# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:40:01 2023

@author: kthat
"""

# ADS1 assignement 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def draw_barplot(df_m):
    """ 
    This method is used to create a bar plot. 

    """

    fig, ax = plt.subplots()
    ax = df_m.plot.bar()
    

    # Add label and title
    ax.set_ylabel('CO2 Emission', fontsize=6)
    ax.set_title('CO2 Emission by Construction Industry',
                 color="red", fontsize=10)

    ax.legend()

    # Save as png image
    plt.savefig("bar_plot.png")
    plt.show()

    return


def print_dataframe(df_m):
    """
    This method is common print used to print data after manipulated dataframe
    """
    x=np.random.randint(1, 100)
    filename = "dataframe" + str(x) + ".txt"
    textfile = open(filename, "w")
    df_m.to_string(textfile)
    textfile.close()
    

# Read data file
data1 = pd.read_excel('C:\\Users\\kthat\\OneDrive\\ADS1\\assignment2\\CO2_emmision_by_Construction.xlsx')

# Create dataframe - remove unnecessary columns and null columns
df = pd.DataFrame(data1)
df = df.head(10).drop(columns=['Country Code','Series Name','Series Code'])

# Manipulate dataframes
df_years_in_column = df.copy()
df_years_in_column.index = df_years_in_column['Country Name']
df_years_in_column = df_years_in_column.iloc[:, 1:-1]

# Create dataframe with countries in column using transpose, and manipulate
df_countries_in_column = df.copy()
df_countries_in_column = df_countries_in_column.transpose()
df_countries_in_column.columns = df_countries_in_column.iloc[0]
df_countries_in_column = df_countries_in_column[1:]
df_countries_in_column.index.names = ["Year"]

# Print basic dataset
print_dataframe(df_years_in_column)
print_dataframe(df_countries_in_column)

# Call plot function
draw_barplot(df_years_in_column)


