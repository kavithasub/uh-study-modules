# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:45:04 2023

@author: kthat
"""
import pandas as pd
dictionary = {"Category" : ["Array", "Stack", "Queue"],
             "Marks":[20, 21, 19]}
df = pd.DataFrame(dictionary)
print(df)

dictionary3 = {"Category" : ["Array", "Stack", "Queue"],
             "Student_1":[20, 21, 19], "Student_2":[15, 12, 20]}
df3 = pd.DataFrame(dictionary3)
print(df3)

dictionary2 = [["Max", 10], ["Bingo", 15], ["Lea", 20]]
df2 = pd.DataFrame(dictionary2, columns=["Name", "Age"])
print(df2)

#dictionary4 = {"Category": ["DS", "Algo"], "Name":["Linked_list", "Stack", "Queue"], "Marks":[10, 9, 7]],
                          # ["Algo", "Name":["Greedy", "DP", "Backtrack"], "Marks":[8, 6, 5]]}




