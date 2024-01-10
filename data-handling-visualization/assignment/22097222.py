# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:13:14 2024

@author: kthat
Data Source: 
https://databank.worldbank.org/source/education-statistics-%5e-all-indicators

Data file: https://github.com/kavithasub/uh-study-modules/tree/main/data-handling-visualization/assignment/Data_Extract_From_Education_Statistics_All_Indicators.xlsx

Github: https://github.com/kavithasub/uh-study-modules/tree/main/data-handling-visualization/assignment
"""
import pandas as pd
import matplotlib.pyplot as plt


def read_data(datafile):
    """
    Read data from csv file and return dataframe
    """
    data = pd.read_excel(datafile)
    return data


def manipulate_data(df, ind_header, ind_value, years, countries):
    """
    Filter data, manipulate, and transpose dataframe to draw plots from it
    """
    # Groupby indicator and values
    df_m = df.groupby(ind_header, group_keys=True)
    df_m = df_m.get_group(ind_value)
    # Set index and filter data
    df_m = df_m.reset_index()
    df_m = df_m.set_index('Country Name')
    df_m = df_m.loc[:, years]
    df_m = df_m.loc[countries, :]
    # Remove null values
    df_m = df_m.dropna()
    # Round to 3 decimal float
    df_m = round(df_m, 2)

    return df_m


def manipulate_data_pie(df, country, year, indicators):
    # Groupby indicator and values
    df_m = df.groupby('Country Name', group_keys=True)
    df_m = df_m.get_group(country)
    df_m = df_m[df_m['Series'].isin(indicators)]

    return df_m


"""
Step1: Read world bank data from file
"""
filename = 'Data_Extract_From_Education_Statistics_All_Indicators.xlsx'
dataframe = read_data(filename)
dataframe = dataframe.fillna(0)

"""
Step2 : Prepare data with manipulating dataframe
"""
# prepare data for bar chart
indicator_header = 'Series'
indicator_value_1 = 'Government expenditure on education as % of GDP (%)'
years = ['2005', '2015']
countries = ['Russia', 'Canada', 'India',
             'Brazil', 'South Africa', 'Spain', 'Mexico', 'Australia']
title_1 = 'Government Expenditure on Education'

df_education_1 = manipulate_data(
    dataframe, indicator_header, indicator_value_1, years, countries)


# prepare data for pie chart
indicators = ['Enrolment in primary education, both sexes (number)',
              'Enrolment in lower secondary education, both sexes (number)',
              'Enrolment in pre-primary education, both sexes (number)',
              'Enrolment in secondary education, both sexes (number)',
              'Enrolment in tertiary education, all programmes, both sexes (number)',
              'Enrolment in upper secondary education, both sexes (number)']
year = '2015'
country = 'World'
title_2 = 'Enrolment Status % Worldwide: 2015'
labels = ['primary', 'lower secondary', 'pre-primary',
          'secondary', 'tertiary', 'upper secondary']
df_education_2 = manipulate_data_pie(dataframe, country, year, indicators)


# prepare data for line plot
indicator_value_2 = 'Enrolment in primary education, both sexes (number)'
years_2 = ['1990', '1995', '2000', '2005', '2010', '2015']
countries_2 = ['World', 'Indonesia', 'India', 'Russia', 'China']
title_3 = 'Primary Education Enrolment Growth'
df_education_3 = manipulate_data(
    dataframe, indicator_header, indicator_value_2, years_2, countries_2)

# Add new column to get enrolment in millions
count = len(years_2)
million = 1000000.00
for i in range(0, count):
    new_column = years_2[i]+'.'
    df_education_3[new_column] = df_education_3[years_2[i]]/million
    df_education_3 = df_education_3.drop(years_2[i], axis=1)

# Use transposed dataframe to plot
df_transpose = df_education_3.transpose()


# prepare data for bar plot 2
indicator_value_3 = 'Gross attendance ratio for tertiary education, both sexes (%)'
countries_3 = ['United States', 'Canada', 'Russia', 'Brazil', 'Mexico']
years_3 = ['2015']
title_4 = 'Gross Attendance Ratio for Tertiary Education'

df_education_4 = manipulate_data(
    dataframe, indicator_header, indicator_value_3, years_3, countries_3)

# Use transposed dataframe to plot
df_transpose_2 = df_education_4.transpose()
header_values = list(df_transpose_2.columns.values)
raw_values = list(df_transpose_2.iloc[0])


"""
Step3 : Ploting graphs
"""
# create subplots
fig = plt.figure(constrained_layout=True, figsize=(10, 10))

fig.suptitle("GLOBAL EDUCATION STATUS: 2015", fontsize=18,
             color='mediumblue', fontweight='bold')

gspec = fig.add_gridspec(ncols=2, nrows=4)

# Plot BAR chart as subplot 1
ax0 = fig.add_subplot(gspec[0, 0])
df_education_1.plot.bar(ax=ax0)
ax0.set_title(title_1, color="red", fontsize=14, fontweight='bold')
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.legend(loc="upper left", fontsize='x-small')
plt.xticks(rotation=-25)
plt.ylabel('Expenditure % of GDP')
plt.subplots_adjust(bottom=0.5, top=0.8)


# Plot PIE chart as subplot 2
ax1 = fig.add_subplot(gspec[1, 0])
plt.pie(df_education_2['2015'], autopct='%1.2f%%',
        textprops=dict(color='w', fontsize='8', fontweight='bold'))
ax1.set_title(title_2, color="red", fontsize=14, fontweight='bold')
ax1.legend(labels=labels, loc='upper left',
           fontsize='x-small', bbox_to_anchor=(1, 1))


# Plot LINE graph as subplot 3
ax2 = fig.add_subplot(gspec[0, 1])
df_transpose.plot.line(ax=ax2, linestyle='solid', linewidth=2)
ax2.set_title(title_3, fontsize=14, color='red', fontweight='bold')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.grid(color='gray', linestyle='--')
ax2.legend(fontsize='x-small', bbox_to_anchor=(1, 1))
plt.ylabel('Enrolment (millions)')
plt.xlabel('Year')
plt.xlim()


# plot BARH graph as subplot 4
ax3 = fig.add_subplot(gspec[1, 1])
# df_transpose_2.plot.barh(ax=ax3)
ax3.barh(header_values, raw_values, color=[
         'purple', 'pink', 'cyan', 'gold', 'lime'])
ax3.set_title(title_4, fontsize=14, color='red', fontweight='bold')
# get rid of the frame
for spine in plt.gca().spines.values():
    spine.set_visible(False)
# remove x-label ticks
plt.xticks([])
# add labels for each bar
for count, cap_data in enumerate(raw_values):
    plt.text(x=cap_data+1, y=count,
             s=f"{cap_data}", color='black', va='center', fontweight='bold')
ax3.legend()

# Add description
ax4 = fig.add_subplot(gspec[2, :])
ax4.set_title("Description", fontsize=12, color='red', fontweight='bold')

paragraph = ("This dashboard analyse the Global Education status in year 2015 "
             "among countries around the world. \n\t The comparision for government "
             "expenditure % of GDP on 2005 and 2015 (upper left) shows that "
             "all countrie's government had continuesly invested on education sector "
             "hence, this report study extended to enrolment & attendance status in 2015 globally.\n"
             "\t Graph(bottom left) shows higher pri-primary enrolement ratio 31.33% whereas "
             "secondary and primary ratio are next high values.\n"
             "\t According to line graph(upper right) the primary education enrolment "
             "growth has increased dramatically worldwide and India also having increases."
             "\n\t United States had higher attendance ratio for tertiary education "
             "comparing to other countries and Brazil also had more than 50 attendance ratio"
             "(bottom right).\n\t Overall, this study shows that the education investment, "
             "students enrolment and attendance are had positive growth globally.")

ax4.text(0.0, 1.0, paragraph, size=13, rotation=0., style='normal', multialignment='left', va='top',
         wrap=True, bbox=dict(boxstyle="round", ec=(1.0, 0.5, 0.5), fc=('khaki')))
ax4.axis('off')

# Add student ID and name
student = ("Student ID: 22097222 \n"
           "Name: Kavitha Subramaniyam")
ax5 = fig.add_subplot(gspec[3, 0])
ax5.text(0.0, 0.1, student, fontsize=13, style='italic', color='b')
ax5.axis('off')

# Save image as png
#plt.savefig('22097222.png', dpi=300)
plt.show()
