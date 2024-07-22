# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:10:43 2024

@author: paula
"""
import numpy as np
import numpy.polynomial.polynomial as pol

def horner_p2(x,p):
    Y= np.zeros(2)
    
    Q1=p[::2]
    Q2=p[1::2]
    
    Y[0]=pol.polyval(x**2,Q1)
    Y[1]=pol.polyval(x**2,Q2)
    
    v=Y[0]+x*Y[1]
    return Y,v

def horner_p3(x,p):
    Y=np.zeros(3)
    
    Q1=p[::3]
    Q2=p[1::3]
    Q3=p[2::3]
    
    Y[0]=pol.polyval(x**3,Q1)
    Y[1]=pol.polyval(x**3,Q2)
    Y[2]=pol.polyval(x**3,Q3)
    
    v=Y[0]+x*Y[1]+(x**2)*Y[2]
    return Y,v

def horner_p(x,p,nt):
    if(nt>len(p)):
        return "The number of threads must be less than " +str(len(p))
    Y=np.zeros(nt)
    v=0
    
    for i in range(0,nt,1):
        Q=p[i::nt]
        Y[i]=pol.polyval(x**nt,Q)
        v+=(x**i)*Y[i]
    return Y,v

for i in range(3):
    if i == 0: # polynomial 1
        p = np.array([1.,2,1])
        x = 1.
    if i == 1: # polynomial 2
        p = np.array([1., -1, 2, -3, 5, -2])
        x = 2.
    if i == 2: # polynomial 3
        p = np.array([1., -1, -1, 1, -1, 0, -1, 1])
        x = -1.
    print('\n----------- Polynomial',str(i+1),' ----------- ')
    print('P, x')
    print(p, x,'\n')
    print(pol.polyval(x,p))
    print('horner_p2 ', horner_p2(x,p))
    print('horner_p3 ',horner_p3(x,p))
    print('horner_p 3',horner_p(x,p,3))
    print('horner_p 4',horner_p(x,p,4))
    print('horner_p 5',horner_p(x,p,5))