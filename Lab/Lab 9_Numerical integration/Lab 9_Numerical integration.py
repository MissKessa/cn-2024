# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:07:51 2024

@author: UO294067
"""

#%% Exercise 1
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol
import sympy as sym

def plotting(f,a,b,nodes):
    xp=np.linspace(a,b)
    #Interpoaltion polynomial
    p=pol.polyfit(nodes, f(nodes), len(nodes)-1)
    yp=pol.polyval(xp,p)
    pa=pol.polyval(a,p) #or yp[0]
    pb=pol.polyval(b,p) #or yp[-1]
    
    plt.figure()
    #Exact area
    plt.plot(xp,f(xp), 'b', label='Exact area')
    plt.plot([a,a,b,b],[f(a),0,0,f(b)],'b')
    
    #Interpolation points
    plt.plot(nodes,f(nodes),'ro', label='Interpolation points')
    
    #Approximate area
    plt.plot(xp,yp,'r--', label='Approximate area')
    plt.plot([a,a,b,b],[pa,0,0,pb],'r--')
    
    plt.legend()
    plt.show()

#First example
f=lambda x:np.exp(x)
a=0.;b=3.
nodes=np.array([1,2,2.5])
plotting(f, a, b, nodes)

#Second example
f=lambda x: np.cos(x)+1.5
a=-3.; b=3.
nodes=np.array([-3.,-1.,0,1.,3.])
plotting(f, a, b, nodes)

#%% Exercise 1a
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol
import sympy as sym

def midpoint(f,a,b,pr=False):
    Ia=(b-a)*f((a+b)/2.)
    
    if pr:
        nodes=np.array([(a+b)/2.])
        plotting(f,a,b,nodes)
    return Ia

#Exact integral
x = sym.Symbol('x', real=True) 
f_sym = sym.log(x)
Ie = sym.integrate(f_sym,(x,1,3))
Ie=float(Ie)

#Approximate integral
f=lambda x:np.log(x)
a=1.;b=3.

#midpoint rule
Ia=midpoint(f,a,b,True)
print('The approximate value is', Ia)
print('The exact value is', Ie)

#%% Exercise 1b
def trapez(f,a,b,pr=False):
    Ia=((b-a)/2)*(f(a)+f(b))
    
    if pr:
        nodes=np.array([a,b])
        plotting(f,a,b,nodes)
    return Ia

#Exact integral
x = sym.Symbol('x', real=True) 
f_sym = sym.log(x)
Ie = sym.integrate(f_sym,(x,1,3))
Ie=float(Ie)

#Approximate integral
f=lambda x:np.log(x)
a=1.;b=3.

Ia=trapez(f,a,b,True)
print('The approximate value is', Ia)
print('The exact value is', Ie)

#%% Exercise 1c
def simpson(f,a,b,pr=False):
    Ia=((b-a)/6.)*(f(a)+4*f((a+b)/2.)+f(b))
    
    if pr:
        nodes=np.array([a,(a+b)/2.,b])
        plotting(f,a,b,nodes)
    return Ia

#Exact integral
x = sym.Symbol('x', real=True) 
f_sym = sym.log(x)
Ie = sym.integrate(f_sym,(x,1,3))
Ie=float(Ie)

#Approximate integral
f=lambda x:np.log(x)
a=1.;b=3.

Ia=simpson(f,a,b,True)
print('The approximate value is', Ia)
print('The exact value is', Ie)

#%% Exercise 1d
def com_plotting(f,a,b,nodes,i):
    xp=np.linspace(a,b)
    #Interpoaltion polynomial
    p=pol.polyfit(nodes, f(nodes), len(nodes)-1)
    yp=pol.polyval(xp,p)
    pa=pol.polyval(a,p) #or yp[0]
    pb=pol.polyval(b,p) #or yp[-1]
    
    if (i==0):
        #Exact area
        plt.plot(xp,f(xp), 'b', label='Exact area')
        
        #Interpolation points
        plt.plot(nodes,f(nodes),'ro', label='Interpolation points')
        
        #Approximate area
        plt.plot(xp,yp,'r--', label='Approximate area')
    else:
        #Exact area
        plt.plot(xp,f(xp), 'b')
        
        
        #Interpolation points
        plt.plot(nodes,f(nodes),'ro')
        
        #Approximate area
        plt.plot(xp,yp,'r--')
    
    plt.plot([a,a,b,b],[pa,0,0,pb],'r--')
    
    

def com_midpoint(f,a,b,n):
    x=np.linspace(a,b,n+1) #divide in subintervals
    Ia=0

    plt.figure()
    plt.plot([a,a,b,b],[f(a),0,0,f(b)],'b')
    for i in range(n):
        Ia+=midpoint(f,x[i],x[i+1])
        nodes=np.array([(x[i]+x[i+1])/2.])
        com_plotting(f,x[i],x[i+1],nodes,i)
    
    
    plt.legend()
    plt.show()
    return Ia

#Exact integral
x = sym.Symbol('x', real=True) 
f_sym = sym.log(x)
Ie = sym.integrate(f_sym,(x,1,3))
Ie=float(Ie)

#Approximate integral
f=lambda x:np.log(x)
a=1.;b=3.

Ia=com_midpoint(f,a,b,5)
print('The approximate value is', Ia)
print('The exact value is', Ie)

#%% Exercise 1e
def com_trapz(f,a,b,n):
    x=np.linspace(a,b,n+1) #divide in subintervals
    Ia=0

    plt.figure()
    plt.plot([a,a,b,b],[f(a),0,0,f(b)],'b')
    for i in range(n):
        Ia+=trapez(f,x[i],x[i+1])
        nodes=np.array([x[i],x[i+1]])
        com_plotting(f,x[i],x[i+1],nodes,i)

    plt.legend()
    plt.show()
    return Ia

#Exact integral
x = sym.Symbol('x', real=True) 
f_sym = sym.log(x)
Ie = sym.integrate(f_sym,(x,1,3))
Ie=float(Ie)

#Approximate integral
f=lambda x:np.log(x)
a=1.;b=3.

Ia=com_trapz(f,a,b,4)
print('The approximate value is', Ia)
print('The exact value is', Ie)

#%% Exercise 1f
def comp_simpson(f,a,b,n):
    x=np.linspace(a,b,n+1) #divide in subintervals
    Ia=0

    plt.figure()
    plt.plot([a,a,b,b],[f(a),0,0,f(b)],'b')
    for i in range(n):
        Ia+=simpson(f,x[i],x[i+1])
        nodes=np.array([x[i],(x[i]+x[i+1])/2.,x[i+1]])
        com_plotting(f,x[i],x[i+1],nodes,i)

    plt.legend()
    plt.show()
    return Ia

#Exact integral
x = sym.Symbol('x', real=True) 
f_sym = sym.log(x)
Ie = sym.integrate(f_sym,(x,1,3))
Ie=float(Ie)

#Approximate integral
f=lambda x:np.log(x)
a=1.;b=3.

Ia=comp_simpson(f,a,b,4)
print('The approximate value is', Ia)
print('The exact value is', Ie)

#%% Exercise 2
def gauss(f,a,b,n):
    [x, w] = np.polynomial.legendre.leggauss(n)
    Ia=0
    nodes =np.zeros(n)

    for i in range(n):
        y=((b-a)/2.)*x[i]+((a+b)/2.)
        Ia+=w[i]*f(y)
        nodes[i]=y
    
    Ia=Ia*((b-a)/2.)
    plotting(f,a,b,nodes)
    return Ia

#Exact integral
x = sym.Symbol('x', real=True) 
f_sym = sym.log(x)
Ie = sym.integrate(f_sym,(x,1,3))
Ie=float(Ie)

#Approximate integral
f=lambda x:np.log(x)
a=1.;b=3.

Ia=gauss(f,a,b,1)
print('1 node')
print('The approximate value is', Ia)
print('The exact value is', Ie)

Ia=gauss(f,a,b,2)
print('2 node')
print('The approximate value is', Ia)
print('The exact value is', Ie)

Ia=gauss(f,a,b,3)
print('3 node')
print('The approximate value is', Ia)
print('The exact value is', Ie)
    
#%% Exercise 3
def netwon_cotes(f,a,b,n):
    if n==1:
        return midpoint(f,a,b)
    elif n==2:
        return trapez(f, a, b)
    elif n==3:
        return simpson(f, a, b)

def degree_of_precision(formula,n):
    error=0
    counter=0
    
    while(error>=0):
        f = lambda x: x**counter
        x = sym.Symbol('x', real=True) 
        f_sym = x**counter
        Ie = sym.integrate(f_sym,(x,1,3))
        Ie=float(Ie)
        error=formula(f,1,3,n)-Ie
        
        if error < 10**(-10):
            error=0.
            
        print('f(x) = x ^'+str(counter)+ ' error = '+str(error))
        counter+=1
        
    degree=counter-2
    print('The degree of precision is'+str(degree))
    return degree

print('----  Midpoint rule (n = 1) ----')
degree_of_precision(netwon_cotes,1)

print('----  Trapezoidal rule (n = 2) ----')
degree_of_precision(netwon_cotes,2)

print('----  Simpson rule (n = 3) ----')
degree_of_precision(netwon_cotes,3)
