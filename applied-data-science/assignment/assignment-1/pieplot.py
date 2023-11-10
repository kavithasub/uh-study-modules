# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:50:22 2023

@author: kthat
"""
import matplotlib.pyplot as plt
import pandas as pd


def draw_pieplot(df):
    """ 
        This method is used to create a pie plot. Arguments:
        A dataframe with total population with HIV from each country taken as y
    """

    plt.figure()
    plot = df.plot.pie(y="Number_Of_People", figsize=(
        8, 8), autopct='%1.2f%%', textprops=dict(color='w'))

    # Add label and title
    plot.set_ylabel("Total People")
    plt.title("Sum of people with HIV for past ~20 years(2000-2022)", fontsize=18)
    plt.legend(loc="upper right")

    # Save as png image
    plt.savefig("pie_plot.png")
    plt.show()

    return


# Read data file
data = pd.read_csv(
    "C:\\Users\\kthat\\OneDrive\\data\\number_people_with_HIV.csv")

# Create data frame
df = pd.DataFrame(data)
print(df)

# Replace NAN values from number_of_people column with zeros
df["Number_Of_People"] = df["Number_Of_People"].fillna(0)

# Get total number of people for each country through out given years
df_manipulate = df.groupby(["Country"]).sum(["Number_Of_People"])
print(df_manipulate)
# Call function
draw_pieplot(df_manipulate)
