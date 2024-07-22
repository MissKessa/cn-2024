#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 1
"""
import numpy as np

def incrementalSearch(f,a,b,n):
    x = np.linspace(a,b,n+1)
    intervals = np.zeros((n,2))
    c = 0
    
    for i in range(n):
        if f(x[i]) * f(x[i+1]) < 0:
            intervals[c] = [x[i], x[i+1]]
            c += 1
     
    return intervals[:c,:]   
#-----------------------------------------
f1 = lambda x: x**5 - 3 * x**2 + 1.6
a1 = -1.; b1 = 1.5; n1 = 25

print('\nIntervals that contain f1 zeros\n') 
print(incrementalSearch(f1,a1,b1,n1)) 
#-----------------------------------------
f2 = lambda x: (x+2) * np.cos(2*x) 
a2 = 0.; b2 = 10.; n2 = 100

print('\nIntervals that contain f2 zeros\n') 
print(incrementalSearch(f2,a2,b2,n2)) 

   
