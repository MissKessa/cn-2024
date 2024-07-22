#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 3b
"""
import numpy as np
from L3_ex1 import horner 
np.set_printoptions(suppress = True)

def polDer(x0, p):
    n = len(p)
    d = np.zeros(n)
    
    q = np.copy(p)
    fact = 1.
    for i in range(n):
        q, r = horner(x0, q)
        d[i] = r * fact
        fact *= i+1
 
    return d 

#-------------------------------------


p = np.array([1., -1, 2, -3,  5, -2])
x0 = 1.
print('Derivatives of P in x0 = 1')
print(polDer(x0,p))

r = np.array([1., -1, -1, 1, -1, 0, -1, 1])
x1 = -1.
print('\nDerivatives of R in x1 = -1')
print(polDer(x1,r))


