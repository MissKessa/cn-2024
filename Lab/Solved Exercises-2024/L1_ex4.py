#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 4
"""
import numpy as np

a = np.arange(1.,4)

#%%
b = np.copy(a)
b = np.append(b,0)
b = b[::-1]
b = np.append(b,0)
b = b[::-1]

print('\n1.')
print('b = ',b)
#%%
b = np.zeros(5)
b[1:-1] = a
print('\n2.')
print('b = ',b)
#%%
c = np.array([0.])
b = np.concatenate((c,a,c))

print('\n3.')
print('b = ',b)