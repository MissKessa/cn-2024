# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:10:01 2024

@author: UO294067
"""

#%% Exercise 1
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def secant (f,x0,x1,tol=1e-6, maxiter=100):
    xk= np.zeros(maxiter)
    xk[0]=x0
    xk[1]=x1
    c=0
    for i in range (2, maxiter+1,1):
        xk[i]=xk[i-1]-f(xk[i-1])*((xk[i-1]-xk[i-2])/(f(xk[i-1])-f(xk[i-2])))
        c=i
            
        if (np.abs(xk[i]-xk[i-1])<tol):
            break
            
    return xk[c],c-1
    

f= lambda x: x**5 - 3*x**2 + 1.6
r=np.zeros(3)

x0=-0.7; x1=-0.6
r[0], i=secant (f,x0,x1)
print(r[0],i)

x0=0.8; x1=0.9
r[1],i=secant(f,x0,x1)
print(r[1],i)

x0=1.2; x1=1.3
r[2],i=secant(f,x0,x1)
print(r[2],i)

#%% Theory 2: Netwon & Bisection with scipy.optimize
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

f  = lambda x: np.sin(x) - 0.1*x - 0.1
df = lambda x: np.cos(x) - 0.1

x = np.linspace(-1,20,1000)

plt.figure(figsize=(15,4))
plt.plot(x,f(x))
plt.plot(x,0*x,'k-')
plt.show()

#We obtain the zeros estimating initial guesses over this plot
x0 = np.array([0., 2., 7., 8.])
raiz = op.newton(f,x0,fprime=df,tol=1.e-6,maxiter=100)       
print(raiz)

#If we do not use the derivative as input argument, the function uses the Secant method
x0 = np.array([0., 2., 7., 8.])
raiz = op.newton(f,x0,tol=1.e-6,maxiter=100)       
print(raiz)


#For Bisection, we need to estimate the initial interva
x0 = np.array([0., 2., 7., 8.])
x1 = x0 + 1
raiz = np.zeros_like(x0)
for i in range(len(raiz)):
    raiz[i]= op.bisect(f,x0[i],x1[i],xtol=1.e-6,maxiter=100)       
    print(raiz[i])

#Let us check the solutions graphically
x = np.linspace(-1,9)

plt.figure()
plt.plot(x,f(x),x,x*0,'k',raiz,raiz*0,'ro')
plt.show()

#%% Exercise 2
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

x = sym.Symbol('x', real=True)
f1_sim = (x**2) + sym.ln(2*x+7)*sym.cos(3*x)+0.1
df1_sim=sym.diff(f1_sim,x)
df2_sim=sym.diff(df1_sim,x)
df3_sim=sym.diff(df2_sim,x)

f1   = sym.lambdify([x], f1_sim,'numpy') 
df1  = sym.lambdify([x], df1_sim,'numpy') 
df2  = sym.lambdify([x], df2_sim,'numpy') 
df3  = sym.lambdify([x], df3_sim,'numpy') 


x = np.linspace(-1,3)              # define the mesh in (-1,1.5)
OX = 0*x                             # define X axis

plt.figure()
plt.plot(x,df1(x)) 
plt.title("Derivate of f")                  
plt.plot(x,OX,'k-')     
plt.show()


x0 = np.array([-1., 0., 1., 2.3, 2.8])
raiz = op.newton(df1,x0,fprime=df2,tol=1.e-6,maxiter=100)  

minima = raiz[::2]
maxima = raiz[1::2]

x = np.linspace(-2,4)

plt.figure()
plt.plot(x,f1(x)) 
plt.title("f")  
plt.plot(minima,f1(minima),'go')     
plt.plot(maxima,f1(maxima),'ro')                
plt.plot(x,OX,'k-')     
plt.show()

x0 = np.array([-0.5, 0.5, 1.5, 2.5, 3.8])
raiz = op.newton(df2,x0,fprime=df3,tol=1.e-6,maxiter=100) 

x = np.linspace(-2,4)
OX = 0*x 

plt.figure()
plt.plot(x,df2(x)) 
plt.title("Second derivate of f")                  
plt.plot(x,OX,'k-')     
plt.show()

plt.figure()
plt.plot(x,f1(x)) 
plt.title("f")  
plt.plot(raiz,f1(raiz),'bo')                     
plt.plot(x,OX,'k-')     
plt.show()


#%% Theory 3: Fixed point
f = lambda x: x**3 - 2*x**2 +1
g = lambda x: - np.sqrt((x**3+1)/2)

a = -1.; b = 0;
x = np.linspace(a, b, 200)       # vector with 200 equally spaced points

plt.figure()
plt.plot(x, f(x),'g-',label='f')              
plt.plot(x, g(x),'r-',label='g')
plt.plot(x, 0*x,'k-')            # OX axis  
plt.plot(x, x,'b-',label='y = x')

r = -0.61803
plt.plot(r,0,'ro',label='zero')     
plt.plot(r,r,'bo',label='fixed point')

plt.legend(loc='best')
plt.show()

a = -0.8; b = -0.3;
x = np.linspace(a, b) 

plt.figure()
plt.plot(x, g(x),'r-', label='g')
plt.plot(x, x, 'b-',label='y = x')
plt.plot([a,b],[b,a],'k-') # Draw the other diagonal

pf = -0.61803
plt.plot(pf,pf,'bo',label='fixed point') 

plt.axis([a, b, a, b])     # Graph in [a,b]x[a,b]
plt.legend(loc='best')
plt.show()

#%%Exercise 3
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def fixedPoint(g,x0,tol=1e-6,maxiter=200):
    