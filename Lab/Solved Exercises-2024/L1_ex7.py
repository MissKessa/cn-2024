#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 7
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi, 2*np.pi,500)
f = lambda x: x * np.sin(3*x)
OX = x*0

plt.figure()
plt.plot(x,f(x))                   # Plot the function
plt.plot(x,OX,'k-')                # Plot X axis
plt.xlabel('x')
plt.ylabel('y')
plt.title('x sin(3x)')
plt.show()

