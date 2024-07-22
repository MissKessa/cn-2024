# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:10:00 2024

@author: uo294067
"""

#%% Horner algorithm
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

p = np.array([1., -1, 2, -3,  5, -2])
p0, p1, p2, p3, p4, p5 = p #P(x)= 1 -x +2^2 -3x^3 +5x^4 -2x^5


g = lambda x: p0 + p1*x + p2*x**2 + p3*x**3 + p4*x**4 + p5*x**5 #P(x)
d1g = lambda x: p1 + 2*p2*x + 3*p3*x**2 + 4*p4*x**3 + 5*p5*x**4 #derivative 1
d2g = lambda x: 2*p2 + 6*p3*x + 12*p4*x**2 + 20*p5*x**3 #derivative 2

#Plot them
a = 0.; b = 1.
x = np.linspace(a,b)
plt.plot(x,0*x,'k')
plt.plot(x,g(x), label = 'g')
plt.plot(x,d1g(x), label = 'd1g')
plt.plot(x,d2g(x), label = 'd2g')
plt.legend()
plt.show()

#We can compute the polynomial value at any point:
x0 = -0.5
print('P value at point       ', x0)
print('With polyval:          ', pol.polyval(x0, p)) #Way 1
print('With lambda function g:', g(x0)) #Way 2