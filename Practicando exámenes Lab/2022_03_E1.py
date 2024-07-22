# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:56:26 2024

@author: paula
"""
import numpy as np

def horner_inv(q,r,x0):
    p=np.zeros(len(q)+1)
    
    p[0]=q[0]
    for i in range(1, len(p), 1):
        if(i==len(p)-1):
            p[i]=r-q[i-1]*x0
        else:
            p[i]=q[i]-q[i-1]*x0
    
    return p

x0 = 1.
q0 = np.array([1., 3])
r0 = 4.
print('P0 = ', horner_inv(q0,r0,x0))
# p0 = np.array([1.,2,1])

x1 = 1.
q1 = np.array([ 1.,  0,  2, -1,  4,])
r1 = 2.
print('\nP1 = ', horner_inv(q1,r1,x1))
# p1 = np.array([1., -1, 2, -3,  5, -2])

x2 = -1.
q2 = np.array([ 1., -2,  1,  0, -1,  1, -2,])
r2 = 3.
print('\nP1 = ', horner_inv(q2,r2,x2))
#p2 = np.array([1., -1, -1, 1, -1, 0, -1, 1])

def pol_from_der(der,x0):
    fact=1
    for i in range (0, len(der),1):
        der[i]/=fact
        fact*=i+1
    
    p=np.array([der[-1]])
    
    for i in range(1,len(der)): 
        r = der[-1-i]
        p = horner_inv(p,r,x0)
    return p

x0 = 1.
d0 = np.array([  2., 6, 14, 48, 96, 120])
print('P0 = ', pol_from_der(d0,x0))
# p = np.array([1., -1, 2, -3,  5, -2])

x1 = -1.
d1 = np.array([3., 0, -34, 240, -1056, 3120, -5760, 5040])
print('P1 = ',pol_from_der(d1,x1))
# r = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    