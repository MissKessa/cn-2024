#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 1
"""
import numpy as np
np.set_printoptions(precision = 2)   # only two decimals
np.set_printoptions(suppress = True) # do not use exponential format

def triangular(A,b):
    m, n = A.shape
    At = np.copy(A)
    bt = np.copy(b)
    
    for k in range(n-1):
        f = At[k+1,k] / At[k,k]
        At[k+1,k]   -= f * At[k,k]
        At[k+1,k+1] -= f * At[k,k+1]
        bt[k+1]     -= f * bt[k]
        
    return At, bt
#--------------------------------------
def main():
    print('-------------  DATA  -------------')
    
    n = 7 
    
    A1 = np.diag(np.ones(n))*3
    A2 = np.diag(np.ones(n-1),1) 
    A = A1 + A2 + A2.T 
    
    b = np.arange(n,2*n)*1.
    
    print('A')
    print(A)
    print('b')
    print(b)
    
    print('\n-------  TRIANGULAR SYSTEM -------')
    At, bt = triangular(A,b)
    
    print('At')
    print(At)
    print('bt')
    print(bt)
    
    print('-------------  DATA  -------------')
    
    
    n = 8 
    
    np.random.seed(3)
    A1 = np.diag(np.random.rand(n))
    A2 = np.diag(np.random.rand(n-1),1)
    A = A1 + A2 + A2.T 
    
    b = np.random.rand(n)
    
    print('A')
    print(A)
    print('b')
    print(b)
    
    print('\n-------  TRIANGULAR SYSTEM -------')
    At, bt = triangular(A,b)
    
    print('At')
    print(At)
    print('bt')
    print(bt)

if __name__ == "__main__":
    main()
    