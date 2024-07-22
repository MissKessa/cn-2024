#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import numpy as np
import matplotlib.pyplot as plt

def bisection(f,a,b,tol=1e-6,maxiter=100):
    
    error = np.inf
    niter = 0
    
    while error > tol and niter < maxiter:
        x = (a+b) / 2.
        
        if f(a) * f(x) < 0:
            b = x
        elif f(x) * f(b) < 0:
            a = x
        else:
            return x, niter
        
        error = b - a
        niter += 1
        
    return x, niter
#-----------------------------------------
f = lambda x: x**5 - 3*x**2 + 1.6    
r = np.zeros(3)

a0 = -0.7; b0 = -0.6
r[0], i0  = bisection(f,a0,b0)
print(r[0], i0)

a1 = 0.8; b1 = 0.9
r[1], i1  = bisection(f,a1,b1)
print(r[1], i1)

a2 = 1.2; b2 = 1.3
r[2], i2  = bisection(f,a2,b2)
print(r[2], i2)

x = np.linspace(-1,1.5)
plt.figure()
plt.plot(x,f(x))
plt.plot(x,0*x,'k')
plt.plot(r,r*0,'ro')
plt.show()
#%%
f = lambda x: (x**3 + 1) / (x**2 + 1) * np.cos(x) - 0.2
r = np.zeros(3)

a0 = -2; b0 = -1.5
r[0], i0  = bisection(f,a0,b0)
print('%.5f' % r[0])

a1 = -1.5; b1 = 0
r[1], i1  = bisection(f,a1,b1)
print('%.5f' % r[1])

a2 = 1.; b2 = 2.
r[2], i2  = bisection(f,a2,b2)
print('%.5f' % r[2])

x = np.linspace(-3,3)
plt.figure()
plt.plot(x,f(x))
plt.plot(x,0*x,'k')
plt.plot(r,r*0,'ro')
plt.show()
