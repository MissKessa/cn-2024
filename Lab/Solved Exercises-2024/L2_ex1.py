#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exxercise 1
"""
import numpy as np

f = lambda x: np.exp(x)
x0 = - 0.4
polynomial = 0.
factorial = 1.
term = np.inf
i = 0


while np.abs(term) > 1.e-8 and i < 100:
    term = x0**i / factorial
    polynomial += term
    factorial *= i+1
    
    i += 1
    
print('Function value in -0.4      = ', f(-0.4))
print('Approximation value in -0.4 = ', polynomial)
print('Number of iterations        = ', i)
    

