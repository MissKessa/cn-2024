#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trapecios
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad



def create_nodes_borders(a,b,n):
    np.random.seed(100)
    nodes = np.random.rand(n)*(b-a) + a
    nodes = np.sort(nodes)
    
    borders=np.zeros(n+1)
    borders[0]=a
    borders[-1]=b
    for i in range(1,len(borders)-1):
        borders[i]=(nodes[i]+nodes[i-1])/2
    return nodes,borders

def approximate_integral(f,a,b,n):
   Ie = quad(f,a,b)[0]
   nodes,borders=create_nodes_borders(a,b,n)
   Ia=0
   for i in range(1, len(borders)):
       base=borders[i]-borders[i-1]
       height=f(nodes[i-1])
       Ia+=base*height
   return Ia, Ie

def plot(f,a,b,n):
    nodes,borders=create_nodes_borders(a,b,n)
    x=np.linspace(a,b)
    
    plt.figure()
    plt.plot(x,f(x))
    
    
    for i in range(1, len(borders)):
        plt.plot([borders[i-1],borders[i],borders[i],borders[i-1],borders[i-1]],[0,0,f(nodes[i-1]),f(nodes[i-1]),0],'r--')
    
    plt.plot(nodes,f(nodes),'o')
    plt.show()
    
def evaluate(f,a,b,tol=1.e-6):
    Ie=0
    Ia=100
    n=1
    while(np.abs(Ia-Ie)>=tol):
        Ia, Ie = approximate_integral(f,a,b,n)
        n+=1
        
    return Ia,Ie,n
 

#%% 
print('================================================== Example 1')
f = lambda x: np.sin(x)
a = 0.; b = np.pi; n = 10

print('===  Exercise 1')
nodes, borders  = create_nodes_borders(a,b,n)
print('Nodes = ',nodes)
print('Borders = ',borders)

print('\n===  Exercise 2')
Ia, Ie = approximate_integral(f,a,b,n)
print('Ia = ', Ia)
print('Ie = ', Ie) 

print('\n===  Exercise 3')
plot(f,a,b,n)

print('\n===  Exercise 4')
Ia, Ie, n = evaluate(f,a,b,tol=1.e-6)    
print('iterations =',n)
print('Ia = ', Ia)
print('Ie = ', Ie) 


# Nodes =  np.array([0.01482472, 0.38192066, 0.42947642, 0.87452322, 1.33366134, 1.70715697, 1.80670898, 2.1072204,  2.59449295, 2.65394249])
# Borders =  np.array([0., 0.19837269, 0.40569854, 0.65199982, 1.10409228, 1.52040916, 1.75693298, 1.95696469, 2.35085667, 2.62421772, 3.14159265])
#%% 
print('\n\n================================================== Example 2')
f = lambda x: np.cos(x) + 1
a = -3.; b = 0.; n = 4

print('===  Exercise 1')
nodes, borders  = create_nodes_borders(a,b,n)
print('Nodes = ',nodes)
print('Borders = ',borders)

print('\n===  Exercise 2')
Ia, Ie = approximate_integral(f,a,b,n)
print('Ia = ', Ia)
print('Ie = ', Ie) 

print('\n===  Exercise 3')
plot(f,a,b,n)

print('\n===  Exercise 4')
Ia, Ie, n = evaluate(f,a,b,tol=1.e-6)    
print('iterations =',n)
print('Ia = ', Ia)
print('Ie = ', Ie) 
    

# Nodes =  np.array([-2.16489184, -1.72644723, -1.36978517, -0.4656716])
# Borders =  np.array([-3.,,-1.94566954, -1.5481162, -0.91772839, 0.])
#%% 
print('\n\n================================================== Example 3')
f = lambda x: np.cos(4*x) + 2
a = -1.; b = 1.; n = 7

print('===  Exercise 1')
nodes, borders  = create_nodes_borders(a,b,n)
print('Nodes = ',nodes)
print('Borders = ',borders)

print('\n===  Exercise 2')
Ia, Ie = approximate_integral(f,a,b,n)
print('Ia = ', Ia)
print('Ie = ', Ie) 

print('\n===  Exercise 3')
plot(f,a,b,n)

print('\n===  Exercise 4')
Ia, Ie, n = evaluate(f,a,b,tol=1.e-6)    
print('iterations =',n)
print('Ia = ', Ia)
print('Ie = ', Ie) 
    

# Nodes =  np.array([-0.99056229, -0.75686176, -0.44326123, -0.15096482, 0.08680988, 0.34149817, 0.68955226])
# Borders =  np.array([-1., -0.87371202, -0.60006149, -0.29711302, -0.03207747,  0.21415403, 0.51552522, 1.])














