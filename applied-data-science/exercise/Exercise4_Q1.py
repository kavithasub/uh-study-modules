# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:44:02 2023

@author: ks23ach
"""
# Exercise 4
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read from csv
bp = pd.read_csv("BP_ann.csv")
bp_year = bp["year"]
bp_annual_return = bp["ann_return"]

bcs = pd.read_csv("BCS_ann.csv")
bcs_year = bcs["year"]
bcs_annual_return = bcs["ann_return"]

tsco = pd.read_csv("TSCO_ann.csv")
tsco_year = tsco["year"]
tsco_annual_return = tsco["ann_return"]

vod = pd.read_csv("VOD_ann.csv")
vod_year = vod["year"]
vod_annual_return = vod["ann_return"]

plt.figure()
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(2, 2, 1)
plt.hist(bp_annual_return, bins=10)
plt.xlabel("BP")
plt.subplot(2, 2, 2)
plt.hist(bcs_annual_return, bins=10) 
plt.xlabel("BCS")
plt.subplot(2, 2, 3)
plt.hist(tsco_annual_return, bins=10)
plt.xlabel("TSCO")
plt.subplot(2, 2, 4)
plt.hist(vod_annual_return, bins=10)
plt.xlabel("VOD")

plt.legend()
plt.show()

plt.figure(1)
plt.subplot(2, 2, 5)
plt.hist(bp_annual_return)
plt.figure(1)
plt.hist(bcs_annual_return)