# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:07:35 2018

@author: Lily
"""

import numpy as np, scipy as sp
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------

def Input_capacitor():
    d = int(input("Please enter a separation distance for the two capacitor plates \n >>> "))
    l = int(input("Please enter a value for the length of the capacitor plates \n >>> "))
    return d, l
#-----------------------------------------------------------------------------

def Input_General():
    N = int(input("Please enter a value for the dimension of the potential matrix (50 is recommended) \n >>> "))
    n_it = int(input("Please enter a value for the number of iterations (1000 is recommended) \n >>> "))
    I = int(input("Please enter a value for potential \n >>> "))
    return N, n_it, I

#-----------------------------------------------------------------------------

def Potential():
    Input_0 = 'Input_0'
    while Input_0 == 'Input_0':
        Input_1 = input("To examine a point charge press '1' or to examine \
two parallel plate capacitors press '2'. \n >>> ")
        if Input_1 == '1' or Input_1 == '2':
            break
    N, n_it, I = Input_General()
    V = np.zeros((N,N))
    if Input_1 == '2':
        d, l = Input_capacitor()

    for i_full in range(n_it):
        for i in range(N):
            for j in range(N):
                Test = np.abs(V[i,j])

                if Input_1 == '1':
                    x1, y1 = N/2, N/2
                    V[x1,y1] = I

                if Input_1 == '2':
                    if i == int((N/2)+(d/2)) and ((N/2)-(l/2)) <= j <= ((N/2)+(l/2)):
                        V[i,j] = I
                        continue

                    elif i == int((N/2)-(d/2)) and ((N/2)-(l/2)) <= j <= ((N/2)+(l/2)):
                        V[i,j] = -I
                        continue

                if i == 0 and j == 0:
                    V[i,j] = (0.5 * (V[i,j+1] + V[i+1,j]))
                elif i == N-1 and j == 0:
                    V[i,j] = (0.5 * (V[i,j+1] + V[i-1,j]))
                elif i == N-1 and j == N-1:
                    V[i,j] = (0.5 * (V[i,j-1] + V[i-1,j]))
                elif i == 0 and j == N-1:
                    V[i,j] = (0.5 * (V[i,j-1] + V[i+1,j]))

                elif i == 0:
                    V[i,j] = (1/3) * (V[i,j-1] + V[i,j+1] + V[i+1,j])
                elif i == N-1:
                    V[i,j] = (1/3) * (V[i,j-1] + V[i,j+1] + V[i-1,j])
                elif j == 0:
                    V[i,j] = (1/3) * (V[i-1,j] + V[i+1,j] + V[i,j+1])
                elif j == N-1:
                    V[i,j] = (1/3) * (V[i-1,j] + V[i+1,j] + V[i,j-1])

                else:
                    V[i,j] = (1/4) * (V[i-1,j] + V[i+1,j] + V[i,j-1] + V[i,j+1])
        if np.abs(V[i,j]) < 1.001*Test:
            break

    return V, d
#-----------------------------------------------------------------------------

def E_Field(V, d):
    N = len(V)
    E = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            E[i,j] = V[i,j] / d
    return E

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# MENU -----------------------------------------------------------------------
Myinput = '0'
while Myinput != 'q':
    Myinput = input('Part 1: Backward Euler Method for different potentials \
\nPart 2: Diffusion problem \nEnter a choice,"1", "2" or "q" to quit: \n >>> ')
    print ("You entered the choice: ",Myinput)
#-----------------------------------------------------------------------------
# PART ONE -------------------------------------------------------------------
    if Myinput == '1':
        # INTRODUCTION -------------------------------------------------------
        print('-'*60)
        print("You have chosen part (1) \n Backward Euler Method")
        V, d = Potential()
        plt.imshow(V, origin='lower', cmap = 'Spectral')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        Cbar = plt.colorbar()
        Cbar.ax.set_ylabel("Potential (V)")
        plt.show()
        plt.imshow(E_Field(V, d), origin='lower', cmap = 'Spectral')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        Cbar = plt.colorbar()
        Cbar.ax.set_ylabel("Electric Field (V/m)")
        plt.show()

#-----------------------------------------------------------------------------
# PART TWO -------------------------------------------------------------------
    if Myinput == '2':
        # INTRODUCTION -------------------------------------------------------
        print('-'*60)
        print("You have chosen part (2) \n Diffusion problem")

#-----------------------------------------------------------------------------

print ("You have chosen to quit.\nGoodbye.")
