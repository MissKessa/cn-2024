# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:03:39 2024

@author: uo294067
"""

#%% Exercise 1
import numpy as np
import matplotlib.pyplot as plt

def incrementalSearch(f,a,b,n):
    x=(b-a)/n #length intervals
    intervals = np.zeros((n,2))
    c=0
    min=a
    max=a+x
    
    for i in range (0, n, 1):
        if(max > b):
            break
        
        if(f(min)*f(max)<0):
            intervals[c, 0]=min
            intervals[c,1]= max
            c+=1
        min+=x
        max+=x
    
    intervals=intervals[:c,:]
    return intervals[:c,:]

f1 = lambda x : x**5 - 3 * x**2 + 1.6
f2 = lambda x : (x+2)* np.cos(2*x)

intervals_f1=incrementalSearch(f1, -1, 1.5, 25)
intervals_f2=incrementalSearch(f2, 0, 10, 100)

#%% Exercise 2
import numpy as np
import matplotlib.pyplot as plt

def bisection(f,a,b,tol=1e-6,maxiter=100):
    ak=a
    bk=b
    xk= np.zeros(maxiter)
    c=0
    for i in range (1, maxiter+1,1):
        x=(ak+bk)/2        
        
        if(f(ak)*f(x)<0):
            bk=x  
            
        elif (f(x)*f(bk)<0):
            ak=x
            
        else:
            break
        
        xk[i-1]=x
        c=i
        
        if(i>1):
            if (np.abs(xk[i-1]-xk[i-2])<tol):
                break    
    return x, c

f1 = lambda x : x**5 - 3 * x**2 + 1.6
x = np.linspace(-1,1.5)              # define the mesh in (-1,1.5)
OX = 0*x                             # define X axis

r=np.zeros(3)

a=-0.7; b=-0.6
r[0], niter=bisection(f1,a,b)
print(r[0],niter)

a=0.8; b=0.9
r[1], niter=bisection(f1,a,b)
print(r[1],niter)

a=1.2; b=1.3
r[2], niter=bisection(f1,a,b)
print(r[2],niter)

plt.figure()
plt.plot(x,f1(x))                   
plt.plot(x,OX,'k-')     
plt.plot(r,r*0,'ro')
plt.show() 

np.set_printoptions(precision = 5)
f2 = lambda x: ((x**3+1)/(x**2+1))* np.cos(x) - 0.2
x = np.linspace(-3,3.5) 
OX = 0*x

inter=incrementalSearch(f2, -3, 3, 25)
r[0], niter=bisection(f2,inter[0,0],inter[0,1])
print(r[0],niter)

r[1], niter=bisection(f2,inter[1,0],inter[1,1])
print(r[1],niter)

r[2], niter=bisection(f2,inter[2,0],inter[2,1])
print(r[2],niter)


plt.figure()
plt.plot(x,f2(x))                   
plt.plot(x,OX,'k-')     
plt.plot(r,r*0,'ro')
plt.show()

#%%Exercise 3
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def newton(f,df,x0,tol=1e-6,maxiter=100):
    xk= np.zeros(maxiter)
    xk[0]=x0
    c=0
    for i in range (1, maxiter+1,1):
        xk[i]=xk[i-1]-(f(xk[i-1])/df(xk[i-1]))
        
        
        c=i
        
        if (np.abs(xk[i]-xk[i-1])<tol):
            break
        
    return xk[c],c

#f1 = lambda x : x**5 - 3 * x**2 + 1.6

x = sym.Symbol('x', real=True)
f1_sim = (x**5) - (3 * x**2) + 1.6
df1_sim=sym.diff(f1_sim,x)

f1   = sym.lambdify([x], f1_sim,'numpy') 
df1  = sym.lambdify([x], df1_sim,'numpy') 

x = np.linspace(-1,1.5)              # define the mesh in (-1,1.5)
OX = 0*x                             # define X axis

r=np.zeros(3)

a=-0.7;
r[0], niter=newton(f1,df1,a)
print(r[0],niter)

a=0.8
r[1], niter=newton(f1,df1,a)
print(r[1],niter)

a=1.2
r[2], niter=newton(f1,df1,a)
print(r[2],niter)

plt.figure()
plt.plot(x,f1(x))                   
plt.plot(x,OX,'k-')     
plt.plot(r,r*0,'ro')
plt.show() 

np.set_printoptions(precision = 5)
x = sym.Symbol('x', real=True)
f2_sim = ((x**3+1)/(x**2+1))* sym.cos(x) - 0.2
df2_sim =sym.diff(f2_sim,x)

f2   = sym.lambdify([x], f2_sim,'numpy') 
df2  = sym.lambdify([x], df2_sim,'numpy') 

x = np.linspace(-3,3.5) 
OX = 0*x

inter=incrementalSearch(f2, -3, 3, 25)
r[0], niter=newton(f2,df2,inter[0,0])
print(r[0],niter)

r[1], niter=newton(f2,df2,inter[1,0])
print(r[1],niter)

r[2], niter=newton(f2,df2,inter[2,0])
print(r[2],niter)


plt.figure()
plt.plot(x,f2(x))                   
plt.plot(x,OX,'k-')     
plt.plot(r,r*0,'ro')
plt.show()

#%% Exercise 4
import numpy as np
import matplotlib.pyplot as plt

def zeros_bisec(f,a,b):
    intervals=incrementalSearch(f,a,b)
    m=len(intervals)
    r = np.zeros(m)
    for
