# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:08:50 2024

@author: paula
"""
import numpy as np
import numpy.polynomial.polynomial as pol

def gridsearch(f1,f2,ax,bx,ay,by,step):
    f= lambda x,y: np.abs(f1(x,y))+np.abs(f2(x,y))
  
    xi=0
    yi=0
    mini=np.inf
    for i in np.arange(ax,bx+step/2, step):
        for j in np.arange(ay, by+step/2, step):
            if f(i,j)<mini:
                mini=f(i,j)
                xi=i
                yi=j
    print('\nIntersection point')
    print('%.2f %.2f'%(xi,yi))

f1 = lambda x,y: (x-1)**2 + y**2 - 5
f2 = lambda x,y: x**2 + (y-0.1)**2 - 3.2
ax=-2.
bx=0.
by=-1.
ay=-2.
step=0.01

gridsearch(f1, f2, ax, bx, ay, by, step)

ax=-0.55
bx=0.
by=2.
ay=1.

gridsearch(f1, f2, ax, bx, ay, by, step)

def Newton(f, x):
    f