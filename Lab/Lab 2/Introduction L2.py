# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:08:32 2024

@author: uo294067
"""

import numpy as np
import time

f=lambda x:np.exp(x) #We define the function f(x)=e^x as lambda function
x=np.linspace(-1,1,300000) #We create a vector with 300000 equispaced elements in the interval [−10,10]
#%% First way creating vector: y grows inside the loop. Worst performance
y=np.array([]) #We start with a numpy array of size zero
t=time.time()
for i in range(len(x)):
    y=np.append(y,f(x[i])) #We will add elements with np.append

t1=time.time()-t

#%% Second way creating vector: reserve space and fill inside the loop. Ok performance
y=np.zeros_like(x) #same form as x but filled it with 0s
t=time.time()
for i in range(len(x)):
    y[i]=f(x[i]) #we fill the elements one by one with the loop
t2=time.time()-t

#%%Third way creating vector: vectorization (numpy way).  Best performance
t=time.time()
y=f(x)
t3=time.time()-t

#%%Plot and save with just a line
import numpy as np
import matplotlib.pyplot as plt

f=lambda x:np.exp(x)
#x=np.linspace(-1,1) same as x=np.linspace(-1,1,50)
x=np.linspace(-1,1,5)
y=f(x)

plt.figure()
plt.plot(x,y) #First way: just a line
plt.saveig('plot.png')#save the imae
plt.show() #Necessary to store it as an image

#%% Plot with dots
import numpy as np
import matplotlib.pyplot as plt

f=lambda x:np.exp(x)
#x=np.linspace(-1,1) same as x=np.linspace(-1,1,50)
x=np.linspace(-1,1,5)
y=f(x)

plt.figure()
plt.plot(x,y,'o-') #Second way: line with dots
plt.show() #Necessary to store it as an image

#%% Plot labels, titles and legends
import numpy as np
import matplotlib.pyplot as plt

f=lambda x:np.exp(x)
#x=np.linspace(-1,1) same as x=np.linspace(-1,1,50)
x=np.linspace(-1,1,5)
y=f(x)

plt.figure()
plt.plot(x,y,label ='function') #Adding label to the graph
plt.plot(x,0*x, 'k', label='Axis') #Draw at black, and Adding label to the axis
plt.title('Example function f plot') #Adding title
plt.legend() #To see labels
plt.show() #Necessary to store it as an image

#%%Taylor series as function
def P(x0,degree):
    polynomial = 0.
    factorial = 1.

    for i in range(degree + 1):
        term = x0**i/factorial
        polynomial += term
        factorial *= i+1
        
    return polynomial

print('P(0.5, 10)  = ', P(0.5, 10))
print('np.exp(0.5) = ', np.exp(0.5))

#If the input argument is a numpy array, the output is also a numpy array
a = -1.; b = 1.
x = np.linspace(a,b,5)
print('x         = ', x)
print('P(x, 10)  = ', P(x, 10))
print('np.exp(x) = ', np.exp(x))

#%%Plotting Taylor polynomial and f
import numpy as np
import matplotlib.pyplot as plt
a = -1.; b = 1.
f = lambda x: np.exp(x)
x = np.linspace(a,b)
OX = 0*x    

plt.figure()
plt.plot(x,f(x), label = 'f')
plt.plot(x,OX,'k')
plt.plot(x,P(x,2),'r', label = 'P2')
plt.title('Example of function and polynomial graph')
plt.legend()                           
plt.show()

#%%Plotting many Taylor polynomials and f
import numpy as np
import matplotlib.pyplot as plt
a = -3.; b = 3.
f = lambda x: np.exp(x)
x = np.linspace(a,b)
OX = 0*x    

plt.figure()
plt.plot(x,f(x), label = 'f')
plt.plot(x,OX,'k') 

for degree in range(1,5):
    plt.plot(x,P(x,degree), label = 'P'+str(degree))
    plt.title('Function and approximation polynomials') 
    plt.legend()                           
    plt.pause(1) #Waits a bit so it creates like an animation
plt.show()
