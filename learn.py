# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:37:58 2023

@author: ks23ach
"""
# TRY TO LEARN ALL PYTHON #

import pandas as pd

ex = pd.read_csv(r'C:\Users\ks23ach\ads_data\test.txt')

message = 'It\'s also .....'

greeting = 'Good '
time = 'Afternoon'
x = 1

greeting = greeting + time + '!'
print(greeting)
print(greeting + "gggg" + str(x))

count = 10_000_000_000
print(count)

print('a' < 'b')

# input #
#price = input('Enter the price ($):')
#tax = input('Enter the tax rate (%):')
#net = int(price) * int(tax) / 100
#print('net price is :' + str(net))

#print(f'net price is {net}') ##### use to print var

# append #
numbers = [1, 3, 2, 7, 9, 4, 9]
numbers.append(10)
numbers.insert(2, 100)
del numbers[0]
print(numbers)
rem = numbers.pop(2)
print(rem, numbers)
numbers.remove(9)
print(numbers)



