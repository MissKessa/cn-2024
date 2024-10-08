#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 5
"""
import numpy as np

A = np.array([[2,1,3,4],
              [9,8,5,7],
              [6,-1,-2,-8],
              [-5,-7,-9,-6]])

print('A =')
print(A)

a = A[:,0]
print('\na =')
print(a)

b = A[2,:]
print('\nb =')
print(b)

c = A[:2,:2]
print('\nc =')
print(c)

d = A[2:,2:]
print('\nd =')
print(d)

e = A[1:3,1:3]
print('\ne =')
print(e)

f = A[:,1:]
print('\nf =')
print(f)

g = A[1:,1:-1]
print('\ng =')
print(g)