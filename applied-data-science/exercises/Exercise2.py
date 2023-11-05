# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:56:31 2023

@author: ks23ach
"""
# Exercises – Plotting and style check

import numpy as np
import matplotlib.pyplot as plt


def hyper() :
    """produce hyperbolic functions"""
    f1 = np.sinh(arr1)
    f2 = np.cosh(arr1)
    return f1,f2

def trigon1(a, b, j) :
    """calculate trigonometric polynomial"""
    f3 = np.cos(a*arr2) - np.cos(b*arr2)**j
    return f3

def trigon2(c, d, k) :
    f4 = np.sin(c * arr2) - np.sin(d * arr2)**k
    return f4
    
    
arr1 = np.linspace(-5, 5, 1000)
arr2 = np.linspace(0, 2*np.pi, 1000)

# question 1
sinh, cosh = hyper()

plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.subplot(2, 2, 1)
plt.plot(arr1, sinh, label="sinh(x)")
plt.plot(arr1,cosh, label="cosh(x)")
plt.legend(loc="upper right")
plt.xlim(-5, 5)
plt.xlabel("x")
plt.ylabel("f(x)")

# question 2
fun2 = trigon1(1, 60, 3)
plt.subplot(2, 2, 2)
plt.plot(arr2, fun2, label="cos(phi) - cos(60*phi)^3")
plt.legend(loc="upper right")
plt.xlabel("phi")
plt.ylabel("phi fun")
plt.xlim(-0, 2*np.pi)

#question 3
fx3 = trigon1(1, 60, 3)
fy3 = trigon2(1, 120, 4)
plt.subplot(2, 2, 3)
plt.plot(fx3, fy3, label="trigonal2")
plt.xlabel("cos(phi) - cos(60*phi)^3")
plt.ylabel("sin(phi) - sin(phi)^4")

plt.show()
