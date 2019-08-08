# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 20:45:09 2018

@author: Elfring
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylab

x=1
y=1

resF = np.array([[0],[0],[(50*9.81)]])

def M(x,y):
    
    x1 = x
    x2 = x - 45*np.sqrt(3)
    x3 = x - 90*np.sqrt(3)
    
    y1 = y
    y2 = y -45*np.sqrt(3)
    y3 = y
    #z^2 is 49 = constant
    r1 = ((x1**2) + (y1**2) + (49))**0.5
    r2 = ((x2**2) + (y2**2) + (49))**0.5
    r3 = ((x3**2) + (y3**2) + (49))**0.5
    
    M = np.array([[x1/r1,x2/r2,x3/r3] , [y1/r1,y2/r2,y3/r3] , [49/r1,49/r2,49/r3]]) #tensions in 3D
    return M

def T1mag(M):
    T1 = np.dot(np.linalg.inv(M) , resF) # matrix equation
    return sp.linalg.norm(T1)  #finds magnitue of tension

print(T1mag(M(0,0)))

def Tensionmap(k):
    
    T = np.zeros((k,k))
    
    for i in range(k):
        x = (i/k)*(90*np.sqrt(3))
        for j in range(k):
            if y >= 135:
                T[i,j] = 0
            
            if x <= 45*np.sqrt(3):
                if  y >= (j/k)*135:
                    T[i,j] = 0
                if y < (j/k)*135:
                    T[i,j] = T1mag(M(x,y))
                
            if x >= 45*np.sqrt(3):
                if y >= 270*(1-(i/k)):
                    T[i,j] = 0
                if y <= 270*(1-(i/k)):
                    T[i,j] = T1mag(M(x,y))
    
    
                    
    return (T)


plt.imshow(Tensionmap(100))




#N = 50 
#p = 0.00025
#k = (2*np.pi)/(0.6e-6)      
#z = 20     
        
#numpoints = 50        
#delta = 2*0.001 / (numpoints - 1)        
#intensity = np.zeros( (numpoints,numpoints) )
        
#for i in range(numpoints):
            
#    x = -0.001 + i * delta
            
#    for j in range(numpoints):
               
 #       y = -0.001+ j * delta
                
 #   intensity[i,j] = (np.abs(E(x,y,z,p,k,N)))**2
 #   plt.imshow(intensity)
 #   plt.show()




                    
                    
                
                    
                    
                
                
                
            #if ((x-15)/y) < (45*np.sqrt(3))/135:
#           T[i,j] = 0
                
#            if (x-15) > 135 - (y/135)*(45*np.sqrt(3)):
    
               





