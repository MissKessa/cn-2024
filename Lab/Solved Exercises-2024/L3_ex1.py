#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 1
"""
import numpy as np
import numpy.polynomial.polynomial as pol

def horner(x0, p):
    n = len(p)
    q = np.zeros(n)
    
    q[-1] = p[-1]
    for i in range(n-2,-1,-1):
        q[i] = p[i] + q[i+1] * x0
    
    quotient = q[1:]
    remainder = q[0]  
      
    return quotient, remainder

#-------------------------------------

def main():
    p0 = np.array([1.,2,1])
    x0 = 1
    q, r = horner(x0,p0)
    rp   = pol.polyval(x0,p0) 

    print('\nQ coefficients = ', q)
    print('P0(1)        = ', r)
    print('With polyval = ', rp)

    p1 = np.array([1., -1, 2, -3,  5, -2])
    x1 = 1.
    q, r = horner(x1,p1)
    rp   = pol.polyval(x1,p1) 

    print('\nQ coefficients = ', q)
    print('P1(1)        = ', r)
    print('With polyval = ', rp)
    
    p2 = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    x2 = -1.
    q, r = horner(x2,p2)
    rp   = pol.polyval(x2,p2) 

    print('\nQ coefficients = ', q)
    print('P2(-1)       = ', r)
    print('With polyval = ', rp)

if __name__ == "__main__":
    main()
