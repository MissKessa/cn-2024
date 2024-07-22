# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:18:49 2024

@author: uo294067
"""

#%% Exercise 1:
import numpy as np
import numpy.polynomial.polynomial as pol
def horner(x0,p):
    n = len(p)
    q = np.zeros_like(p) #or q = np.zeros(n)
    q[-1]=p[-1]
        
    for i in range (n-2,-1,-1):
        q[i]=p[i]+q[i+1]*x0
    
    remainder=q[0]
    quotient=q[1::]
    return quotient,remainder
    
    
def main():
    p0 = np.array([1.,2,1])
    x0 = 1.
    q, r = horner(x0,p0)
    rp   = pol.polyval(x0,p0) 
    
    print('\nQ coefficients = ', q)
    print('P0(1)        = ', r)
    print('With polyval = ', rp)
    
    
    p0 = np.array([1., -1, 2, -3,  5, -2])
    x0 = 1.
    q, r = horner(x0,p0)
    rp   = pol.polyval(x0,p0) 
    
    print('\nQ coefficients = ', q)
    print('P0(1)        = ', r)
    print('With polyval = ', rp)
    
    p0 = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    x0 = -1.
    q, r = horner(x0,p0)
    rp   = pol.polyval(x0,p0) 
    
    print('\nQ coefficients = ', q)
    print('P0(1)        = ', r)
    print('With polyval = ', rp)

if __name__ == "__main__":
    main()
    
    
#%% Exercise 2
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt
from Homework_3 import horner

def hornerV(x,p):
    y = np.zeros_like(x)
    
    for i in range (0, len(x),1):
        q,y[i]= horner(x[i],p)
    return y

def main():
    x=np.linspace(-1,1)
    p = np.array([1., -1, 2, -3, 5, -2])
    hornerP=hornerV(x,p)
    r = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    hornerR=hornerV(x,r)
    
    plt.figure()
    plt.plot(x,hornerR)
    plt.plot(x,0*x,'k')
    plt.title('hornerR')
    plt.show()
    
    plt.figure()
    plt.plot(x,hornerP)
    plt.plot(x,0*x,'k')
    plt.title('hornerP')
    plt.show()
if __name__ == "__main__":
    main()

#%%Exercise 3
import numpy as np
import numpy.polynomial.polynomial as pol
from Homework_3 import horner

def polDer(x0,p):
    derives = np.zeros_like(p)
    remainders = np.zeros_like(p)
    factorial=1
    for i in range (0, len(p),1):
        p,remainders[i]= horner(x0,p)
        derives[i]=remainders[i]* factorial
        factorial*=(i+1)
        
    return derives, remainders

p = np.array([1., -1, 2, -3,  5, -2])
x0 = 1.

r = np.array([1., -1, -1, 1, -1, 0, -1, 1])
x1 = -1.

dP, rP = polDer(x0,p)
dR, rR = polDer(x1,r)

#%%Exercise 4
def divisors(m):
    m = abs(int(m))
    divisors= np.zeros(2*m)
    n=0
    
    for i in range (1,m,1):
        if (m%i==0):
            