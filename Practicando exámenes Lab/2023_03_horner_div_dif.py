# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:37:11 2024

@author: paula
"""
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt

def divdif(p,z0,h):
    y0 = z0+h
    x0 = z0-h
    
    Py=0
    Px=0
    
    for i in range(0,len(p),1):
        Py+=(y0**i)*p[i]
        Px+=(x0**i)*p[i]
    d=(Py-Px)/(y0-x0)
    return d

def divdifpol(p,z0,h):
    y0 = z0+h
    x0 = z0-h
    
    Py=pol.polyval(y0,p)
    Px=pol.polyval(x0,p)
    d=(Py-Px)/(y0-x0)
    return d

def divdifv(p,z,h):
    d=np.zeros(len(z))
    
    for i in range(0,len(z),1):
       y0 = z[i]+h
       x0 = z[i]-h
       Py=0
       Px=0
       
       for j in range(0,len(p),1):
           Py+=(y0**j)*p[j]
           Px+=(x0**j)*p[j]
       
       d[i]=(Py-Px)/(y0-x0)
       
    return d
    
def derP(p):
    dp=np.zeros(len(p)-1)
    
    for i in range(1,len(p)):
        dp[i-1]=i*p[i]
    return dp

def plotP(p,a,b):
    x=np.linspace(a,b,100)
    
    dP=divdifv(p,x,0.001)
    derivP=derP(p)
    
    plt.figure()
    plt.plot(x,pol.polyval(x,p),'c', label='P with polyval')
    plt.plot(x,dP,'y', label='dP with differences')
    plt.plot(x,pol.polyval(x,derivP),'k--', label='dP with polyval')
    plt.legend()
    plt.show()

print('\n----------------  Exercise 1  ----------------')
p1 = np.array([1., 5, 0, -1, 2, -3, 1])
z1 = 1.1
h1 = 0.001
print('P1 = ',p1, 'at z1 = ', z1, 'with h1 = ',h1)
print('\nWith the proposed algorithtm ', divdif(p1,z1,h1))
print('With polyval                 ', divdifpol(p1,z1,h1))

p2 = np.array([5., 1, -1, 2, -3])
z2 = 0.1
h2 = 0.0005
print('\nP2 = ',p2, 'at z2 = ', z2, 'with h2 = ',h2)
print('\nWith the proposed algorithtm ', divdif(p2,z2,h2))
print('With polyval                 ', divdifpol(p2,z2,h2))

print('\n----------------  Exercise 2  ----------------')
p1 = np.array([1., 5, 0, -1, 2, -3, 1])
z1 = np.arange(0,1,0.2)
h1 = 0.001
print('Values of P1 = ',p1,'at',z1)
print('\n', divdifv(p1,z1,h1))

p2 = np.array([5., 1, -1, 2, -3])
z2 = np.arange(-1,1,0.4)
h2 = 0.01
print('\nValues of P2 = ',p2,'at',z2)
print('\n', divdifv(p2,z2,h2))

print('\n----------------  Exercise 3  ----------------')
p1 = np.array([1., 5, 0, -1, 2, -3, 1])
print('Derivative of P1 = ',p1)
print('\ndP1 = ',derP(p1))

p2 = np.array([5., 1, -1, 2, -3])
print('\nDerivative of P2 = ',p2)
print('\ndP2 = ',derP(p2))

print('\n----------------  Exercise 4  ----------------')
p1 = np.array([1., 5, 0, -1, 2, -3, 1])
a1 = 0.; b1 = 2.
print('P1 = ',p1,'in (',a1,',',b1,')')
plotP(p1,a1,b1)

p2 = np.array([5., 1, -1, 2, -3])
a2 = -0.5; b2 = 0.5
print('P2 = ',p2,'in (',a2,',',b2,')')
plotP(p2,a2,b2)