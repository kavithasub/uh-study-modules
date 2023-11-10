# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:36:27 2023

@author: kthat
"""

import pandas as pd
import matplotlib.pyplot as plt


def draw_barplot(df_m):
    """ 
    This method is used to create a bar plot. Arguments:
    A dataframe with a column 'Period' (used for years) taken as x and 
    other columns taken as y. List 'country' containing column to plot.
    """

    fig, ax = plt.subplots()
    ax.bar(df_m["Country"], df_m["ValueAccurate"],
           width=0.5, color="limegreen")

    # Add label and title
    ax.set_ylabel('Propotion of Population (%)')
    ax.set_title('Population with primary reliance on fuels and technologies',
                 color="indigo", fontsize=15)

    # Save as png image
    plt.savefig("bar_plot")
    plt.show()

    return


# Read data file
data = pd.read_csv(
    "C:\\Users\\kthat\\OneDrive\\data\\popuation-cleanfuels-technology2.csv")

# Create data frame
df = pd.DataFrame(data)

# Remove column 'ParentLocation' which is not useful
df = df.drop(columns=["ParentLocation", "IsLatestYear", "CountryCode"])

# Rename column 'FactValueNumeric' to '' for easy understanding
df = df.rename(columns={"FactValueNumeric": "ValueAccurate"})

# Call function
draw_barplot(df)
