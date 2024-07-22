#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import numpy as np
from L4_ex1 import triangular
np.set_printoptions(precision = 2)   # only two decimals
np.set_printoptions(suppress = True) # do not use exponential format


def back_subs(At,bt):
    n = len(bt)
    x = np.zeros(n)
    
    x[n-1] = bt[n-1] / At[n-1,n-1]
    for k in range(n-2,-1,-1):
        x[k] = (bt[k] - At[k,k+1] * x[k+1]) / At[k,k]
    
    return x    
        
#-------------------------- 
# Example 1
n = 7 

A1 = np.diag(np.ones(n))*3
A2 = np.diag(np.ones(n-1),1) 
A = A1 + A2 + A2.T 

b = np.arange(n,2*n)*1. 

At, bt  = triangular(A,b)
x = back_subs(At,bt)   

print('x')
print(x) 
#-------------------------- 
# Example 2
n = 8 

np.random.seed(3)
A1 = np.diag(np.random.rand(n))
A2 = np.diag(np.random.rand(n-1),1)
A = A1 + A2 + A2.T 

b = np.random.rand(n) 

At, bt  = triangular(A,b)
x = back_subs(At,bt)   

print('x')
print(x)