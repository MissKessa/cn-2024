# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:08:45 2024

@author: UO294067
"""

#%% Exer 1
import numpy as np
np.set_printoptions(precision = 8)   
np.set_printoptions(suppress = True)

def jacobi(A,b,tol= 1.e-6,maxiter=1000):
    x=np.zeros(len(A)) #Current iteration
    xp = np.copy(x) #Previous iteration
    numiter=1
    for k in range(0,maxiter):
        for i in range (0, len(x),1):
            sum1=0
            sum2=0
            for j in range (0, i):
                sum1+=A[i][j]*xp[j]
            for j in range (1+i,len(x)):
                sum2+=A[i][j]*xp[j]
                
            x[i]=(1/A[i][i]) * (b[i]-sum1-sum2)
            
        if np.linalg.norm(x-xp) < tol:
            break
       
        xp = np.copy(x)
        numiter+=1
        
    return x, numiter

A = np.array([[5.,1,-1,-1],[1,4,-1,1],[1,1,-5,-1],[1,1,1,-4]])
b = np.array([1.,1,1,1])
xs=np.linalg.solve(A,b)
x,numiter=jacobi(A,b)

print('Exact x: ', xs)
print('Approximate x: ', x)
print('Iterations: ', numiter)

n = 9 
A1 = np.diag(np.ones(n))*2 
A2 = np.diag(np.ones(n-1),1)
A = A1 + A2 + A2.T 
b = np.concatenate((np.arange(1.,6),np.arange(4,0,-1)))
xs=np.linalg.solve(A,b)
x,numiter=jacobi(A,b)
print('Exact x: ', xs)
print('Approximate x: ', x)
print('Iterations: ', numiter)

#%%Exer 2
def gauss_seidel(A,b,tol= 1.e-6,maxiter=1000):
    x=np.zeros(len(A)) #Current iteration
    xp = np.copy(x) #Previous iteration
    numiter=1
    for k in range(0,maxiter):
        for i in range (0, len(x),1):
            sum1=0
            sum2=0
            for j in range (0, i):
                sum1+=A[i][j]*x[j] #Change xp to x
            for j in range (1+i,len(x)):
                sum2+=A[i][j]*xp[j]
                
            x[i]=(1/A[i][i]) * (b[i]-sum1-sum2)
            
        if np.linalg.norm(x-xp) < tol:
            break
       
        xp = np.copy(x)
        numiter+=1
        
    return x, numiter

A = np.array([[5.,1,-1,-1],[1,4,-1,1],[1,1,-5,-1],[1,1,1,-4]])
b = np.array([1.,1,1,1])
xs=np.linalg.solve(A,b)
x,numiter=gauss_seidel(A,b)

print('Exact x: ', xs)
print('Approximate x: ', x)
print('Iterations: ', numiter)

n = 9 
A1 = np.diag(np.ones(n))*2 
A2 = np.diag(np.ones(n-1),1)
A = A1 + A2 + A2.T 
b = np.concatenate((np.arange(1.,6),np.arange(4,0,-1)))
xs=np.linalg.solve(A,b)
x,numiter=gauss_seidel(A,b)
print('Exact x: ', xs)
print('Approximate x: ', x)
print('Iterations: ', numiter)

#%%Exer 3
def SOR(A,b,w,tol= 1.e-6,maxiter=1000):
    x=np.zeros(len(A)) #Current iteration
    xp = np.copy(x) #Previous iteration
    numiter=1
    for k in range(0,maxiter):
        for i in range (0, len(x),1):
            sum1=0
            sum2=0
            for j in range (0, i):
                sum1+=A[i][j]*x[j] #Change xp to x
            for j in range (1+i,len(x)):
                sum2+=A[i][j]*xp[j]
                
            x[i]=(w/A[i][i]) * (b[i]-sum1-sum2) + (1-w)*xp[i]
            
        if np.linalg.norm(x-xp) < tol:
            break
       
        xp = np.copy(x)
        numiter+=1
        
    return x, numiter

n = 9 
A1 = np.diag(np.ones(n))*2 
A2 = np.diag(np.ones(n-1),1)
A = A1 + A2 + A2.T 
b = np.concatenate((np.arange(1.,6),np.arange(4,0,-1)))
xs=np.linalg.solve(A,b)
x,numiter=SOR(A,b,1.5)
print('Exact x: ', xs)
print('Approximate x: ', x)
print('Iterations: ', numiter)