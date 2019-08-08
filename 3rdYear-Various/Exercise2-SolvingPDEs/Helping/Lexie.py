# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:57:17 2018

@author: Lexie Lodge
"""

import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

def LUD(A,C,N):

    """Uses the LUD method to solve simultaneous equations. Takes a matrix (A),
    an array (C) and number of unknowns (N) as its arguments."""

    LUD_1=sp.linalg.lu_factor(A)
    LUD_2=sp.linalg.lu_solve(LUD_1,C)

    return LUD_2

def alpha(k,rho,C):

    """Returns a value for thermal diffusivity. Takes themal conductivity (k),
    density (rho) and heat capacity (C) as its arguments."""

    return k/(rho*C)

def backward_euler(L,tmax,N,Nt):

    """Uses the backward Euler method to solve the heat equation in one dimension.
    Takes distance in x (L), time elapsed (tmax), iterations over x (N) and
    iterations over time (t) as arguments."""

    x_vals = np.linspace(0,L,N)
    dx = x_vals[1] - x_vals[0]
    t_vals = np.linspace(0,tmax,Nt)
    dt = t_vals[1] - t_vals[0]
    T_prime = np.zeros(N)
    T = np.array([20]*N)#np.zeros(N)
    A = np.zeros((N,N))

    X = alpha(59,7900,450)*(dt/((dx)**2))

    for i in range(1,N-1):

        A[i,i-1] = -X
        A[i,i+1] = -X
        A[i,i] = 1 + 2*X
        A[0,0] = A[N-1,N-1] = 1

    loop = '0'
    while (loop == '0'):
        option = input("For Part (a) input 'a', for Part (b) (end submerged in ice), input 'b': ")
        if option == 'a' or option == 'b':
            break

    for counter in range(Nt):
        T[0] = 1000

        T_prime = LUD(A,T,N)

        if option == 'a':
            T_prime[-1] = T_prime[-2]

        if option == 'b':
            T_prime[-1] = 0

        T = T_prime

    print(T[N-1])
    return x_vals,T_prime

N=100
Nt=100
tmax=5000
L=0.5

result = backward_euler(L,tmax,N,Nt)

plt.plot(result[0],result[1])
plt.xlabel("Distance Along Poker (m)")
plt.ylabel("Temperature (°C)")
plt.suptitle("Temperature Along Length of a Hot Poker")
plt.show()

"""fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
cmap = mpl.cm.hot
norm = mpl.colors.Normalize(vmin=min(result[0]), vmax=max(result[0]))
mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal')
plt.show()"""
