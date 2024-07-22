# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:11:49 2024

@author: paula
"""
import numpy as np
import numpy.polynomial.polynomial as pol

def divpol(D,d):
    m = len(D)
    n = len(d)
    Qlen = m - n + 1
    Qi=np.zeros(Qlen)
    Di = D[::-1]
    di = d[::-1]
    
    a=Di[:n:]
    b = di*a[0]
    c = a - b
    Qi[0] = a[0]
    
    for i in range (0, Qlen-1,1):
        a=np.append(c[1:])
        a=a.append(Di[n+i])
        
        b = di*a[0]
        c = a - b
        Qi[i+1] = a[0]
    
    Ri=c[1:]
    
    Q=Qi[::-1]
    R = Ri[::-1]
    return Q,R

D = np.array([1.,5,-1,1,0,1])
d = np.array([3.,2,1])
np.random.seed(1)
D = np.random.rand(10)
d = np.random.rand(4)
d[-1] = 1.

def bindiff(a,b):
    Di1=a.append(np.zeros(len(b)-1))
    di=b



