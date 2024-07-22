#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 3
"""
import numpy as np
np.set_printoptions(precision = 2)   # only two decimals
np.set_printoptions(suppress = True) # do not use exponential format

def triangular(Ar,b):
    n = len(b)
    At = np.copy(Ar)
    bt = np.copy(b)
    
    for k in range(n-1):
        f = At[k+1,0] / At[k,1]
        At[k+1,0]   -= f * At[k,1]
        At[k+1,1]   -= f * At[k,2]
        bt[k+1]     -= f * bt[k]
        
    return At, bt 
#--------------------------------------------- 
def back_subs(At,bt):
    n = len(bt)
    x = np.zeros(n)
    
    x[n-1] = bt[n-1] / At[n-1,1]
    for k in range(n-2,-1,-1):
        x[k] = (bt[k] - At[k,2] * x[k+1]) / At[k,1]
    
    return x
#---------------------------------------------  
    

print('-------------  DATA  -------------')
n = 7 

Ar = np.zeros((n,3))
Ar[:,0] = np.concatenate((np.array([0]),np.ones((n-1),)))
Ar[:,1] = np.ones((n),)*3
Ar[:,2] = np.concatenate((np.ones((n-1),),np.array([0])))

b = np.arange(n,2*n)*1.

print('Ar')
print(Ar)
print('b')
print(b)

print('\n-------  TRIANGULAR SYSTEM -------')
At, bt = triangular(Ar,b)

print('At')
print(At)
print('bt')
print(bt)

print('\n-----------  SOLUTION  -----------')

x = back_subs(At,bt)   

print('x')
print(x)

print('-------------  DATA  -------------')
n = 8

np.random.seed(3)
Ar = np.zeros((n,3))
Ar[:,1] = np.random.rand(n)
Ar[:,0] = np.concatenate((np.array([0]),np.random.rand(n-1)))
Ar[0:n-1,2] = Ar[1:n,0]

b = np.random.rand(n)

print('Ar')
print(Ar)
print('b')
print(b)

print('\n-------  TRIANGULAR SYSTEM -------')
At, bt = triangular(Ar,b)

print('At')
print(At)
print('bt')
print(bt)

print('\n-----------  SOLUTION  -----------')

x = back_subs(At,bt)   

print('x')
print(x)