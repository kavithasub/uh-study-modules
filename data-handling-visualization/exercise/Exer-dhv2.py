# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:06:27 2023

@author: kthat
"""
# Exercise 2.2.1
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age' :     [ 10, 22, 13, 21, 12, 11, 17],
    'section' : [ 'A', 'B', 'C', 'B', 'B', 'A', 'A'],
    'city' :    [ 'Gurgaon', 'Delhi', 'Mumbai', 'Delhi', 'Mumbai', 'Delhi', 'Mumbai'],
    'gender' :  [ 'M', 'F', 'F', 'M', 'M', 'M', 'F'],
    'favourite_color' : [ 'red', np.NAN, 'yellow', np.NAN, 'black', 'green', 'red']
})
print(df)

print(df.loc[df.age >= 15])

print(df.loc[(df.age >= 12) & (df.gender == "M")])

#row = df.loc[df.age >= 12]
#print(df.city[row], df.gender[row]) // not work


#df.loc[(df.age >= 12), ["section" == "M"]] // not work
print(df)

print(df.iloc[[0,2], [1,3]])


# A series A of 10 random integers from 0 to 9 and indexed from 0 to 9
A = pd.Series(np.random.randint(10, size=10))
print(A)
# A series B of 10 random integers from 0 to 9 and indexed from 5 to 14
B = pd.Series(np.random.randint(10, size=10), index=range(5, 15))
print(B)
print(A+B)
C = pd.Series(np.random.randint(10, size=10), index=list("abcdefghij"))
print(C)
CD = pd.DataFrame(C["C"], C["D"])
CD["CD"] = CD["C"]*CD["D"] 


