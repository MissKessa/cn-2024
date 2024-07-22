#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 3
"""
import numpy as np
import matplotlib.pyplot as plt

def newton(f,df,x0,tol=1e-6,maxiter=100):
    
    error = np.inf
    niter = 0
    
    while error > tol and niter < maxiter:
        x1 = x0 - f(x0) / df(x0)
        error = np.abs(x1-x0)
        niter += 1
        x0 = x1
        
    return x1, niter
#-----------------------------------------
f = lambda x: x**5 - 3*x**2 + 1.6   
df = lambda x: 5*x**4 - 6*x  
r = np.zeros(3)

x0 = -0.7
r[0], i0  = newton(f,df,x0)
print(r[0], i0)

x1 = 0.8
r[1], i1  = newton(f,df,x1)
print(r[1], i1)

x2 = 1.2
r[2], i2  = newton(f,df,x2)
print(r[2], i2)

x = np.linspace(-1,1.5)
plt.figure()
plt.plot(x,f(x))
plt.plot(x,0*x,'k')
plt.plot(r,r*0,'ro')
plt.show()
#%%
import sympy as sym

x = sym.Symbol('x', real=True)

f_sim   = sym.cos(x)*(x**3+1)/(x**2+1) - 0.2
df_sim  = sym.diff(f_sim,x)

f   = sym.lambdify([x], f_sim,'numpy') 
df  = sym.lambdify([x], df_sim,'numpy') 
#%%
r = np.zeros(3)

x0 = -2
r[0], i0  = newton(f,df,x0)
print('%.5f' % r[0])

x1 = -0.5
r[1], i1  = newton(f,df,x1)
print('%.5f' % r[1])

x2 = 2
r[2], i2  = newton(f,df,x2)
print('%.5f' % r[2])

x = np.linspace(-3,3)
plt.figure()
plt.plot(x,f(x))
plt.plot(x,0*x,'k')
plt.plot(r,r*0,'ro')
plt.show()
