# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:37:31 2023

@author: kthat
"""

import matplotlib.pyplot as plt
import pandas as pd

def draw_lineplot(df, countries):
    """ 
        This method used to create a line plot. Arguments:
        A dataframe with a column 'Period' (used for years) taken as x and 
        other columns taken as y. List 'country' containing column to plot. 
    """
    plt.figure()
    
    for country in countries:
        plt.plot(df["Period"], df[country], label=country, linewidth=2.0)

    plt.annotate("high HIV identified", xy=(2020, 200000), 
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    # Add label
    plt.xlabel("Year")
    plt.ylabel("Number of People")
    # Add title
    plt.title("Number of People with HIV during 2008 - 2022", color='red', fontsize=18)
    
    # Remove white space from left and right
    plt.xlim(min(df["Period"]))  
    
    # Save as image
    plt.savefig("line_plot.png")
    plt.legend(loc="center right", fontsize='x-small')
    plt.show()
    
    return

# Read data file
data = pd.read_csv("C:\\Users\\kthat\\OneDrive\\data\\number_people_with_HIV_sheet2.csv")

#df_hiv = pd.DataFrame(read_data2, columns=("Code", "Country", "Period", "Number_Of_People"))
df = pd.DataFrame(data)

# Replace nan values with zeros
df = df.fillna(0)

# Get column names by excluding first index(Period) column
countries = df.columns[1:]

# Call function
draw_lineplot(df, countries)









