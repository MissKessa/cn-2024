# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:14:52 2024

@author: UO294067
"""

#%% Exercise 1
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

np.set_printoptions(precision = 2)   # only 2 fractionary digits
np.set_printoptions(suppress = True) # do not use exponential notation

def approxi1(f,deg,a,b,n):
    #Build the points
    x=np.linspace(a,b,n)
    y=f(x)
    
    #Build V
    V=np.ones((n,deg+1))
    for i in range (0, n):
        for j in range(1,deg+1):
            V[i][j]=x[i]**j
    #Build the system
    C=V.T@V
    d=np.dot(V.T,y)
    
    #solve
    p=np.linalg.solve(C,d)
    
    x1=x
    x=np.linspace(a,b,50)
    
    plt.figure()
    plt.plot(x1,y,'ro', label='points')
    plt.plot(x,pol.polyval(x,p), label='fitting polynomial')  
    plt.legend()  
    plt.show()
    
    return p

f1 = lambda x: np.sin(x)
deg1=2
n1=5
a1=0
b1=2
poly=approxi1(f1,deg1,a1,b1,n1)



f1 = lambda x: np.cos(np.arctan(x))-np.log(x+5)
deg1=4
n1=10
a1=-2
b1=0
poly=approxi1(f1,deg1,a1,b1,n1)

#%% Exercise 2
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol
import pandas as pd

df=pd.read_csv('http://www.unioviedo.es/compnum/labs/new/cars.csv',sep=',')
x=df['weight']
y=df['horsepower']
z=df['origin']

b=z==1 #boolean variable same size as z (MASK)

plt.figure()
plt.plot(x[z==1], y[z==1],'bo', label='USA')
plt.plot(x[z==2], y[z==2],'yo', label='Europe')
plt.plot(x[z==3], y[z==3],'co', label='Japan')
fit=pol.polyfit(x,y,1)
point=pol.polyval(3000,fit)
plt.plot(x,pol.polyval(x,fit),'r')
plt.plot(3000,point,'ro')
plt.legend()
plt.xlabel('weight')
plt.ylabel('horsepower')
plt.show()


x=df['horsepower']
y=df['mpg']
z=df['origin']

b=z==1 #boolean variable same size as z (MASK)

plt.figure()
plt.plot(x[z==1], y[z==1],'bo', label='USA')
plt.plot(x[z==2], y[z==2],'yo', label='Europe')
plt.plot(x[z==3], y[z==3],'co', label='Japan')
x_sorted=np.sort(x)
fit=pol.polyfit(x,y,2)
point2=pol.polyval(point,fit)
plt.plot(x_sorted,pol.polyval(x_sorted,fit),'r')
plt.plot(point,point2,'ro')
plt.legend()
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.show()

#%% Exercise 3
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol
from scipy.integrate import quad

def approx2(f,deg,a,b): #Vp=y    
    #Build V
    V=np.zeros((deg+1,deg+1))
    for i in range (0, deg+1):
        beg=0
        start=beg
        for j in range(0,deg+1):
            g=lambda x: x**start
            Ia = quad(g,a,b)[0]
            V[i][j]=Ia
            start+=1
        beg+=1
    
    #Build y
    y=np.zeros(deg+1)
    for i in range(0, deg+1):
        g=lambda x: (x**start) *f(x)
        Ia = quad(g,a,b)[0]
        y[i]=Ia
    
    
    #solve
    x=np.linspace(a,b,50)
    #Build the system
    C=V.T@V
    d=np.dot(V.T,y)
    
    #solve
    p=np.linalg.solve(C,d)
    

    
    plt.figure()
    plt.plot(x,f(x),'ro', label='points')
    plt.plot(x,pol.polyval(x,p), label='fitting polynomial')  
    plt.legend()  
    plt.show()
    
    return p
f1 = lambda x: np.sin(x)
deg1=2
n1=5
a1=0
b1=2
poly=approx2(f1,deg1,a1,b1)
