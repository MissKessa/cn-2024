# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:38:19 2024

@author: uo294067
"""

#%% Exercise 1
import numpy as np

def P(x0,degree):
    polynomial = 0.
    factorial = 1.
    tol=(10)**(-8)
    term=tol
    maxNumSum=100
    i=0

    while np.abs(term)>=tol and i<=maxNumSum and i<degree+1:
        term = x0**i/factorial
        polynomial += term
        factorial *= i+1
        i+=1
        
    return polynomial,i


f = lambda x: np.exp(x)

print("Function value in -0.4      =",f(-0.4))
aprox, it = P(-0.4,10)
print("Approximation value in -0.4 =",aprox)
print("Number of iterations =",it)

#%% Exercise 2
import numpy as np
import matplotlib.pyplot as plt

def P(x0,degree, tol, maxNumSum):
    polynomial = 0.
    factorial = 1.
    term=tol
    i=0

    while np.max(np.abs(term))>=tol and i<=maxNumSum and i<degree+1:
        term = x0**i/factorial
        polynomial += term
        factorial *= i+1
        i+=1
        
    return polynomial

def funExp(x, tol, maxNumSum):
    y=np.zeros_like(x)
    for i in range(len(x)):
       y[i]=P(x[i], 10, tol, maxNumSum)
    
    return y
    
x = np.linspace(-1,1)
tol=(10)**(-8)
maxNumSum=100
OX = 0*x

plt.figure()
f = lambda x: np.exp(x)
plt.plot(x,f(x), 'y',linewidth=5, label = 'f')
plt.plot(x,OX,'k') 
plt.plot(x,funExp(x,tol,maxNumSum), 'b--', label = 'f approximation')

plt.title('f approximation with McLaurin seies') 
plt.legend()                           
plt.show()

#%%Exercise 3
import numpy as np

def P(x0,degree):
    polynomial = 0.
    factorial = 1.
    tol=(10)**(-8)
    term=tol
    maxNumSum=100
    positive =True
    i=1
    it=0

    while np.abs(term)>=tol and i<=maxNumSum and it<degree+1:
        term = x0**i/factorial
        if(positive):
            polynomial += term
            positive=False
        else:
            polynomial -= term
            positive=True
        factorial *= i+1
        factorial *= i+2
        i+=2
        it+=1
        
    return polynomial,it


f = lambda x: np.sin(x)

print("Function value      =",f(np.pi/4))
aprox, it = P(np.pi/4,6)
print("Approximation value =",aprox)
print("Number of iterations =",it)

#%%Exercise 4
import numpy as np
import matplotlib.pyplot as plt

def P(x0,degree, tol, maxNumSum):
    polynomial = 0.
    factorial = 1.
    term=tol
    positive =True
    i=1

    while np.max(np.abs(term))>=tol and i<=maxNumSum and i<degree+1:
        term = x0**i/factorial
        if(positive):
            polynomial += term
            positive=False
        else:
            polynomial -= term
            positive=True
        factorial *= i+1
        factorial *= i+2
        i+=2
        
    return polynomial

def funSin(x, tol, maxNumSum):
    y=np.zeros_like(x)
    for i in range(len(x)):
       y[i]=P(x[i], 10, tol, maxNumSum)
    
    return y
    
x = np.linspace(-np.pi,np.pi)
tol=(10)**(-8)
maxNumSum=100
OX = 0*x

plt.figure()
f = lambda x: np.sin(x)
plt.plot(x,f(x), 'y',linewidth=5, label = 'f')
plt.plot(x,OX,'k') 
plt.plot(x,funSin(x,tol,maxNumSum), 'b--', label = 'f approximation')

plt.title('f approximation with McLaurin seies') 
plt.legend()                           
plt.show()

#%%Exercise 5
import numpy as np

def P(x0,degree,i):
    polynomial = 0.
    factorial = 1.
    
    term=0
    #termBefore=np.inf
    maxNumSum=100
    it=0

    while i<=maxNumSum and it<degree+1:
        #if(it>=1):
            #termBefore=term
        
        term = x0**i/factorial
        polynomial += term
        factorial *= i+1
        factorial *= i+2
        i+=2
        it+=1  
        
    return polynomial

def tanha(x0):
    tanhBefore=np.inf
    tanhActual=0
    tol=10**(-4)
    it=0
    
    while (np.abs(tanhActual-tanhBefore)>=tol):
        if(it>0):
            tanhBefore=tanhActual
        tanhActual=P(x0,it,1)/P(x0,it,0)
        it+=1
        
    return tanhActual

f = lambda x: np.tanh(x)

print("Function value      =",f(0.5))
aprox= tanha(0.5)
print("Approximation value =",aprox)

#%%Exercise 6
import numpy as np
import matplotlib.pyplot as plt

def P(x0,degree,i):
    polynomial = 0.
    factorial = 1.
    
    term=0
    #termBefore=np.inf
    maxNumSum=100
    it=0

    while i<=maxNumSum and it<degree+1:
        term = x0**i/factorial
        polynomial += term
        factorial *= i+1
        factorial *= i+2
        i+=2
        it+=1  
        
    return polynomial

def tanha(x0,tol):
    tanhBefore=np.inf
    tanhActual=0
    it=0
    
    while (np.abs(tanhActual-tanhBefore)>=tol):
        if(it>0):
            tanhBefore=tanhActual
        tanhActual=P(x0,it,1)/P(x0,it,0)
        it+=1
        
    return tanhActual

def funTanh(x, tol):
    y=np.zeros_like(x)
    for i in range(len(x)):
       y[i]=tanha(x[i], tol)
    
    return y
    
x = np.linspace(-3,3)
tol=(10)**(-8)
OX = 0*x

plt.figure()
f = lambda x: np.tanh(x)
plt.plot(x,f(x), 'y',linewidth=5, label = 'f')
plt.plot(x,OX,'k') 
plt.plot(x,funTanh(x,tol), 'b--', label = 'f approximation')

plt.title('f approximation with McLaurin seies') 
plt.legend()                           
plt.show()