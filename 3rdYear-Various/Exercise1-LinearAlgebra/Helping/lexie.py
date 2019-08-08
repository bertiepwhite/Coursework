# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:54:35 2018

@author: user
"""
import numpy as np
import scipy as sp
from scipy import linalg
import time
import random

def random_matrix(N):
    
    rand_vals=[]
    for i in range(N**2):
        
        rand_vals.append(random.randint(0,10))
    
    matrix=np.reshape(rand_vals,(N,N))
    
    return matrix

def matrixNxN(vals,N):
    
    if len(vals) == N**2:
        return np.reshape(vals,(N,N))
    else:
        print("The matrix is not NxN")

def detNxN(A,N):
    
    if len(A[0])==2:
        
        n1=A[0][0]
        n2=A[0][1]
        n3=A[1][0]
        n4=A[1][1]
    
        return (n1*n4)-(n2*n3)
    
    else:
        
        det_A=0
 
        for i in range(len(A[0])):
            
            A1=np.delete(A,0,0)
            A2=np.delete(A1,i,1)
            det_A += ((-1) ** i )*A[0][i] * detNxN(A2,len(A2[0]))
            
        return det_A

def cofactors(N):
    
    if N==2:
        
        return np.array([[A[1,1],-A[0,1]],[-A[1,0],A[0][0]]])
    
    else:
    
        sub_dets=[]
        
        for i in range(N):
            for j in range(N):
                
                A1=np.delete(A,i,0)
                A2=np.delete(A1,j,1)
                
                sub_dets.append(((-1)**(i+j))*detNxN(A2,N-1))
                
        return np.reshape(sub_dets,(N,N))


def inverseNxN(A,N):
    
    cofactors_A=cofactors(N)
    det_A=detNxN(A,N)
    
    if det_A != 0:
        return ((1/det_A)*np.transpose(cofactors_A))
    else:
        print("The matrix has no inverse")
    

def inverse_timer(N):
    
    times=[]

    for i in range(N):
        
        A=random_matrix(N)
        start_time = time.time()
        inverseNxN(A,i)
        end_time = time.time() - start_time
        
        times.append(end_time)
    
    return(times)

print(inverse_timer(9))
#N=8
#A=random_matrix(N)
#print(A)
#print(detNxN(A,N))
#print(cofactors(N))
#print(inverseNxN(A,N))