# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:09:49 2024

@author: UO294067
"""

#%%Lagrage polynomial
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

#The points
x = np.array([-1., 0, 2, 3, 5])
y = np.array([ 1., 3, 4, 3, 1])

#plot points and Lagrande polynomials
xp = np.linspace(min(x),max(x))
p  = pol.polyfit(x,y,len(x)-1)
yp = pol.polyval(xp,p)

plt.figure()
plt.plot(xp,yp,'b-', label = 'interpolant polynomial') #Connect the points
plt.plot( x, y,'ro', label = 'points') #Drawing the points
plt.legend()
plt.show()

#%%Exercise 1
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

 
def lagrange_fundamental_v1(k,x,z): #x are the points
    #Calculate lk by skipping the node with position k in x
    #Then calculate the value of lk in z
    enumerator=1
    denominator=1
    for i in range(0, len(x), 1):
        
        if(i!=k):
            enumerator*=z-x[i]
            denominator*=x[k]-x[i]
    
    return enumerator/denominator
    
x = np.array([-1., 0, 2, 3, 5])

k = 2
z = 1.3
yz = lagrange_fundamental_v1(k,x,z)
print(yz)

def lagrange_fundamental(k,x,z): #x are the points
    #Calculate lk by skipping the node with position k in x
    #Then calculate the value of lk in z
    
    result=np.zeros(len(z))
    
    for j in range(0, len(result),1):
        enumerator=1
        denominator=1
        for i in range(0, len(x), 1):
            
            if(i!=k):
                enumerator*=z[j]-x[i]
                denominator*=x[k]-x[i]
        
        result[j]=enumerator/denominator
    
    return result

k = 2
z = np.array([1.3, 2.1, 3.2])
yz = lagrange_fundamental(k,x,z)
print(yz)

#Plotting
x = np.array([-1., 0, 2, 3, 5])
# y = np.array([(1.,0.,0.,0.,0.), (0.,1.,0.,0.,0.), (0.,0.,1.,0.,0.), (0.,0.,0.,1.,0.), (0.,0.,0.,0.,1.)])
h=np.eye(len(x))


#plot points and Lagrande polynomials
for i in range (0, len(h),1):
    y=h[i]
    xp = np.linspace(min(x),max(x))
    # p  = pol.polyfit(x,y,len(x)-1)
    # yp = pol.polyval(xp,p)
    yp=lagrange_fundamental(i,x,xp)

    
    plt.figure()
    plt.plot(xp,yp,'b-', label = 'interpolant polynomial') #Connect the points
    plt.plot( x, y,'ro', label = 'points') #Drawing the points
    plt.title('L'+str(i))
    plt.legend()
    plt.show()
    
#%% Exercise 2
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as pol

def lagrange_polynomial(x,y,z):
    p=np.zeros(len(z))
    for i in range(0, len(y),1):
        p+=y[i]*lagrange_fundamental(i,x,z)
        
    return p

#The points
x = np.array([-1., 0, 2, 3, 5])
y = np.array([ 1., 3, 4, 3, 1])

#plot points and Lagrande polynomials
xp = np.linspace(min(x),max(x))
yp = lagrange_polynomial(x,y,xp)

plt.figure()
plt.plot(xp,yp,'b-', label = 'interpolant polynomial') #Connect the points
plt.plot( x, y,'ro', label = 'points') #Drawing the points
plt.legend()
plt.show()

x = np.array([-1., 0, 2, 3, 5, 6, 7])
y = np.array([ 1., 3, 4, 3, 2, 2, 1])

xp = np.linspace(min(x),max(x))
yp = lagrange_polynomial(x,y,xp)

plt.figure()
plt.plot(xp,yp,'b-', label = 'interpolant polynomial') #Connect the points
plt.plot( x, y,'ro', label = 'points') #Drawing the points
plt.legend()
plt.show()

#%% Chebyshev nodes
x = np.array([-1., 0, 2, 3, 5])
y = np.array([ 1., 3, 4, 3, 1])
p = pol.polyfit(x,y,len(x)-1)  # polynomial coefficients

xp = np.linspace(min(x),max(x))
yp = pol.polyval(xp,p)         # polynomial value in the points contained in xp

plt.figure()
plt.plot(xp, yp, 'b-',x, y, 'ro')
plt.show()

#%% Exercise 3
def chebyshev(f,a,b,n):
    equispaced(f,a,b,n)
    chebynodes(f,a,b,n)
    
def equispaced(f,a,b,n):
    x=np.linspace(a,b,n)
    y=f(x)
    
    #allx=np.linspace(a,b)
    xp = np.linspace(min(x),max(x),200)
    p  = pol.polyfit(x,y,len(x)-1)
    yp = pol.polyval(xp,p)

    
    plt.figure()
    plt.axis([-1.05, 1.05, -0.3, 2.3])
    #plt.plot(allx,f(allx),'b-', label = 'function')
    plt.plot(xp,f(xp),'b-', label = 'function')
    plt.plot(xp,yp,'r-', label = 'interpolant polynomial') #Connect the points
    plt.plot(x, y,'ro', label = 'points') #Drawing the points
    plt.title("Equispaced nodes")
    plt.legend()
    plt.show()

def chebynodes(f,a,b,n):
    x=np.zeros(n)
    
    for i in range (1,n+1,1):
        x[i-1]=np.cos((2*i-1)*np.pi/(2*n))
    
    y=f(x)
    
    #allx=np.linspace(a,b)
    xp = np.linspace(min(x),max(x),200)
    p  = pol.polyfit(x,y,len(x)-1)
    yp = pol.polyval(xp,p)

    
    plt.figure()
    plt.axis([-1.05, 1.05, -0.3, 2.3])
    #plt.plot(allx,f(allx),'b-', label = 'function')
    plt.plot(xp,f(xp),'b-', label = 'function')
    plt.plot(xp,yp,'r-', label = 'interpolant polynomial') #Connect the points
    plt.plot(x, y,'ro', label = 'points') #Drawing the points
    plt.title("Chebyshev nodes")
    plt.legend()
    plt.show()

f = lambda x: 1/(1+25*x**2)
chebyshev(f,-1,1,11)

f = lambda x: np.exp(-20*x**2)
chebyshev(f,-1,1,15)