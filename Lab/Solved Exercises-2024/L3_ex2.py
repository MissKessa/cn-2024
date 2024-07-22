#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2
"""
import numpy as np
import matplotlib.pyplot as plt


def hornerV(x, p):
    n = len(p)
    m = len(x)
    q = np.zeros(n)
    y = np.zeros(m)
    
    for k in range(m):
        q[-1] = p[-1]
        for i in range(n-2,-1,-1):
            q[i] = p[i] + q[i+1] * x[k]
        
        y[k] = q[0]  
      
    return y

#-------------------------------------
def main():
    x = np.linspace(-1,1)
    
    p = np.array([1., -1, 2, -3, 5, -2])
    y = hornerV(x, p)
    
    plt.figure()
    plt.plot(x,y)
    plt.plot(x,0*x,'k')
    plt.title('P')
    plt.show()
    
    r = np.array([1., -1, -1, 1, -1, 0, -1, 1])
    y = hornerV(x, r)
    
    plt.figure()
    plt.plot(x,y)
    plt.plot(x,0*x,'k')
    plt.title('R')
    plt.show()

if __name__ == "__main__":
    main()

