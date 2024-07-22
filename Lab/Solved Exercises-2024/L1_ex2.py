#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import numpy as np
np.set_printoptions(precision=2,suppress=True)
#%%
a = np.arange(7,16,2)
print('a = ', a)

b = np.arange(10,5,-1)
print('b = ', b)

c = np.arange(15,-1,-5)
print('c = ', c)

d = np.arange(0,1.1,0.1)
print('d = ', d)

e = np.arange(-1,1.1,0.2)
print('e = ', e)

f = np.arange(1,2.1,0.1)
print('f = ', f)

#%%
a = np.linspace(7,15,5)
print('\na = ', a)

b = np.linspace(10,6,5)
print('b = ', b)

c = np.linspace(15,0,4)
print('c = ', c)

d = np.linspace(0,1,11)
print('d = ', d)

e = np.linspace(-1,1,11)
print('e = ', e)

f = np.linspace(1,2,11)
print('f = ', f)
