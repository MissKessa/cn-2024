# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:08:04 2024

@author: uo294067
"""

#%%
import numpy as np

n = 7 

A1 = np.diag(np.ones(n))*3
A2 = np.diag(np.ones(n-1),1) 
A = A1 + A2 + A2.T #3-diagonal matrix 7x7
b = np.arange(n,2*n)*1.

print('A')
print(A)
print('b')
print(b)

#Solve the system:
x = np.linalg.solve(A,b)

print('x')
print(x)

#%% Exercise 1
import numpy as np
def triangular (A,b):
    At=np.copy(A)
    bt=np.copy(b)
    m, n = A.shape
    nb = len(b)
    
    if (n!=nb):
        return At,bt
    
    for i in range (0,n-1, 1):
        f=At[i+1,i]/At[i,i]
        
        At[i+1,i]=0
        At[i+1,i+1]=At[i+1,i+1]-f*At[i,i+1]
        bt[i+1]=bt[i+1]-f*bt[i]
    
    return At, bt

np.set_printoptions(precision = 2) 
np.set_printoptions(suppress = True) 

n = 7 

A1 = np.diag(np.ones(n))*3
A2 = np.diag(np.ones(n-1),1) 
A = A1 + A2 + A2.T 

b = np.arange(n,2*n)*1.

At,bt = triangular(A,b)

n = 8 

np.random.seed(3)
A1 = np.diag(np.random.rand(n))
A2 = np.diag(np.random.rand(n-1),1)
A = A1 + A2 + A2.T 

b = np.random.rand(n)
At2,bt2 = triangular(A,b)

#%% Exercise 2
import numpy as np

def back_subs(At,bt):
    x=np.copy(bt)
    n=len(bt)
    
    x[n-1]=bt[n-1]/At[n-1][n-1]
    
    for i in range (n-2,-1, -1):
        x[i]=bt[i]-At[i][i+1]*x[i+1]
        x[i]=x[i]/At[i][i]
    
    return x
       
x1= back_subs(At,bt)

#%% Exercise 3
import numpy as np

def triangular (A,b):
    At=np.copy(A)
    bt=np.copy(b)
    m, n = A.shape
    nb = len(b)
    
    if (n!=nb):
        return At,bt
    
    for i in range (0,n-1, 1):
        f=At[i+1,i]/At[i,i]
        
        At[i+1,i]=0
        At[i+1,i+1]=At[i+1,i+1]-f*At[i,i+1]
        bt[i+1]=bt[i+1]-f*bt[i]
    
    return At, bt

n = 8

np.random.seed(3)
Ar = np.zeros((n,3))
Ar[:,1] = np.random.rand(n)
Ar[:,0] = np.concatenate((np.array([0]),np.random.rand(n-1)))
Ar[0:n-1,2] = Ar[1:n,0]

b = np.random.rand(n)
    
    