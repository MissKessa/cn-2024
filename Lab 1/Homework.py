# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:15:57 2024

@author: uo294067
"""
#%%Exercise 1
import numpy as np

a = np.array([1, 3, 7])
b = np.array([(2, 4, 3), (0, 1, 6)])
c = np.ones(3)
d = np.zeros(4)
e = np.zeros((3, 2))
f = np.ones((3, 4))

#%%Exercise 2
import numpy as np
np.set_printoptions(precision=1,suppress=True)

a = np.arange(7, 16, 2)
b = np.arange(10, 5, -1)
c = np.arange(15, -1, -5)

#as floats
a1 = np.linspace(7, 15, 5)
b1 = np.linspace(10, 6, 5)
c1 = np.linspace(15, 0, 4)

d=np.linspace(0, 1, 11)
e=np.linspace(-1, 1, 10)
f=np.arange(1, 2.1, 0.1)

#%%Exercise 3
import numpy as np
v = np.arange(0., 12.2, 1.1)
vi = v[::-1]

v1=v[::2]
v2=v[1::2]

v11=v[::3]
v21=v[1::3]
v31=v[2::3]

v12=v[::4]
v22=v[1::4]
v321=v[2::4]
v4=v[3::4]

#%%Exercise 4
import numpy as np
a = np.array([1, 2, 3])

b1 = np.append(a,0)
b1 =b1[::-1]
b1 = np.append(b1,0)
b1 =b1[::-1]

b2= np.zeros(5)
b2[1:4:1]=b2[1:4:1]+a

c=np.array([0])
b3=np.concatenate((a,c), axis=None)
b3=np.concatenate((c,b3), axis=None)

#%%Exercise 5
import numpy as np

A=np.array([(2,1,3,4),(9,8,5,7),(6,-1,-2,-8),(-5,-7,-9,-6)])

a=A[:,0]
b=A[2,:]
c=A[0:2:1,0:2:1]
d=A[2::,2::]
e=A[1:3:1,1:3:1]
f=A[::,1::]
g=A[1::,1:3:1]

#%%Exercise 6
import numpy as np

def f1(x):
    return x*np.exp(x)

f=f1(2)

def g1(z):
    return z/(np.sin(z)*np.cos(z))

g=g1(np.pi/4)

def h1(x,y):
    return (x*y)/(x**2+y**2)


h=h1(2,4)

#%%Exercise 7
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2*np.pi,2*np.pi)              # Define the grid x from -6 to 6
f = lambda x : x*np.sin(3*x)   # Define the function

OX = 0*x #To make a line in the drawig at x=0

plt.figure()                       
plt.plot(x,f(x))                   # Plot the function
plt.plot(x,OX,'k-')                # Plot X axis
plt.xlabel('x')
plt.ylabel('y')
plt.title('x sin(3x)')
plt.show()
