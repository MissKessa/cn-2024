"""
EXAM 4
"""
import numpy as np
import numpy.polynomial.polynomial as pol
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt

np.set_printoptions(suppress = True,precision = 5)

def Tmatrix(x0,n):
    T=np.zeros((n,n))
    T[0,0]=x0
    if n==1:  
        return T
    
    for i in range(0, n,1):
            
        for j in range (0, n,1):
                  
            if(i==j):
                if (i!=0 and j!=0):
                    T[i,j]=2*x0
                
                if (i+1<n):
                    T[i+1, j]=1
                if (j+1<n):
                    T[i,j+1]=1
    return T
            
    
#-------------------------------
def de(M):
    #triangularization
    A = M.copy()
    n =len(A)
    for i in range (0,n-1,1):
        f=A[i+1,i]/A[i,i]
        
        A[i+1,i]=0
        A[i+1,i+1]=A[i+1,i+1]-f*A[i,i+1]
            
    #multiply
    result=1
    
    for i in range(0,n,1):
        result*=A[i,i]
    return result
#-------------------------------
def Tch(n): 
    x=np.linspace(-0.99,0.98,n+1) #x0,x1,x2,x3,x4...
    
    A=np.zeros((n+1,n+1))
    
    y=np.zeros(len(x))
    
    for i in range (0, n+1, 1): #A
        for j in range (0,n+1,1):
            A[i,j]=x[i]**j
            
    for i in range(0, len(x),1): #y
        y[i]=de(Tmatrix(x[i],n))
    
    c=np.linalg.solve(A,y)
    return c
#-------------------------------
def plotT(n):
    x=np.linspace(-1,1,300)
    
    
    plt.figure()
    
    for i in range(1,n+1,1):
        c = Tch(i)
        l="T"+str(i)
        plt.plot(x,pol.polyval(x,c),label=l)
        
    plt.legend()
    plt.show()
    
    

#%% EXERCISE 1
print('-------------  EXERCISE 1  -------------')
print('\nTmatrix(2.,1)')
print(Tmatrix(2.,1))
print('\nTmatrix(2.,3)')
print(Tmatrix(2.,3))
print('\nTmatrix(1.,5)')
print(Tmatrix(1.,5))

#%% EXERCISE 2
print('\n-------------  EXERCISE 2  -------------')
print('--- DATA 1 det')
n = 7 
A1 = np.diag(np.ones(n))*3
A2 = np.diag(np.ones(n-1),1) 
A = A1 + A2 + A2.T 
print(de(A))

print('---  DATA 2 det')
n = 8 
np.random.seed(3)
A1 = np.diag(np.random.rand(n))
A2 = np.diag(np.random.rand(n-1),1)
A = A1 + A2 + A2.T 
print(de(A))


print('\nT1(2)    = ', de(Tmatrix(2.,1)))
print('T2(0.61) = ', de(Tmatrix(0.61,2)))
print('T3(2)    = ', de(Tmatrix(2.,3)))
print('T4(-0.5) = ', de(Tmatrix(-0.5,4)))
print('T5(1)    = ',de(Tmatrix(1.,5)))

#%% EXERCISE 3
print('\n-------------  EXERCISE 3  -------------')
print('\nT1 = ',Tch(1))   
print('T2 = ',Tch(2))   
print('T3 = ',Tch(3))   
print('T4 = ',Tch(4)) 
print('T5 = ',Tch(5))
print('T6 = ',Tch(6))  
print('T7 = ',Tch(7))

#%% EXERCISE 4
print('\n-------------  EXERCISE 4  -------------')
# T1 = np.array([0., 1.])
# T2 = np.array([-1.,  0.,  2.])
# T3 = np.array([ 0., -3., -0.,  4.])
# T4 = np.array([ 1.,  0., -8., -0.,  8.])
# T5 = np.array([ -0.,   5.,   0., -20.,  -0.,  16.])
# T6 = np.array([ -1., 0., 18., 0., -48., -0., 32.])
# T7 = np.array([  -0.,-7.,-0., 56.,0.,-112.,-0.,64.])
plotT(3)
plotT(5)

#-------------------------------
'''
from scipy.special import chebyt
n = 5
c = np.array(chebyt(n))
print(c)
'''



