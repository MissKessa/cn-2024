#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 6
"""
import numpy as np

f = lambda x: x * np.exp(x)
print('f(2) = ', f(2))

g = lambda z: z / (np.sin(z) * np.cos(z))
print('g(pi/4) = ', g(np.pi/4))

h = lambda x, y: x * y /(x**2 + y**2)
print('h(2,4) = ', h(2,4))