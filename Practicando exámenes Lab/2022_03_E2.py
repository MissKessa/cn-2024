# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:28:51 2024

@author: paula
"""
import numpy as np

def add_pol(p,q):
    if (len(p)>len(q)):
        n=len(q)
        s=p.copy()
    else:
        n=len(p)
        s=q.copy()
    
    pos=-1
    
    for i in range (n-1, -1,-1):
        s[pos]=p[pos]+q[pos]
        pos-=1
    
    return s

def t2x(p):
    q=2*p
    t=np.zeros(len(p)+1)
    
    for i in range(0, len(p), 1):
        t[i]=q[i]
    return t
    
    
p0 = np.array([1.,2,1])
p1 = np.array([1., -1, 2, -3, 5, -2])
p2 = np.array([1., -1, -1, 1, -1, 0, -1, 1])

print('P1 + P0 = ',add_pol(p1,p0))
print('\nP0 + P1 = ',add_pol(p1,p0))
print('\nP1 + P1 = ',add_pol(p1,p1))
print('\nP1 + P2 = ',add_pol(p1,p2))
print('\nP2 + P1 = ',add_pol(p2,p1))
print('\nP2 + P2 = ',add_pol(p2,p2))

print('\n2x P0 = ',t2x(p0))
print('\n2x P1 = ',t2x(p1))
print('\n2x P2 = ',t2x(p2))


def T(degree):
    t=np.zeros(degree+1)
    if(degree == 0):
        t[0]=1 #1
        return t
    elif (degree==1): #x
        t[1]=1
        t[0]=0
        return t[::-1]
    
    Tn=2*T(degree-1)
    
    for i in range(0, len(Tn)):
        t[len(Tn)-i]=Tn[i]
        
    Tn1=T(degree-2)
    
    for i in range(0, len(Tn1)):
        t[i]+=Tn1[i]
    
    return t

for i in range(0, 9):
    print(str(i),": ",T(i))