# -*- coding: utf-8 -*-
"""
Exercise 5

"""
import pandas as pd

uk_cities = pd.read_csv('C:\\Users\\ks23ach\Desktop\\data\\UK_cities.txt', sep='\s+')
uk_cities = uk_cities.dropna()
print(uk_cities)

#print(uk_cities.describe()) ##gives statistical overview
#print(uk_cities.mean())
#print(uk_cities.median())
#print(uk_cities.sum())
#print(uk_cities.std())
print(uk_cities.corr())
print(uk_cities.corr(method="kendall"))

pop_1000 = uk_cities["Population"] / 1000
#print(pop_1000)
uk_cities["Population_1000"] = pop_1000
#print(uk_cities)
#print(uk_cities["Population_1000"].sum(groupby="Nation/Region"))
print(uk_cities.groupby(["Nation/Region"]))

#help(sum)