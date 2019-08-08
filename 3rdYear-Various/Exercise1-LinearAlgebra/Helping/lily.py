# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:44:23 2018

@author: Lily
"""

import numpy as np, timeit, scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#-----------------------------------------------------------------------------
def Cofactors(z, n):
    """ Calculates the cofactor matrix using input of 'z', the original matrix
    and 'n', the dimension of the matrix 'z'. """

    z_array = []                                    # Empty list for cofactor values

    # If it is a 2x2 Matrix, perform following manual calculation:
    if n == 2:
        return np.array([[z[1][1], - z[1][0]], [-z[0][1],  z[0][0]]] )

    # For nxn matrix where n>2, perform the following calculation:
    for j in range(n):                              # Looping through rows
        for k in range(n):                          # Looping through columns

            z_1 = np.delete(z, j, 0)                # Deletes current row, producing z_1
            z_2 = np.delete(z_1, k, 1)              # Deletes corresponding column from z_1

            z_3 = Det(z_2, len(z_2))                # Calculates the determinant of the z_2 for matrix of minors

            C = (-1)**(j+k) * z_3                   # Relavent signs to change matrix of minors to cofactor matrix
            z_array.append(C)                       # Add these values to the list

    return np.resize(np.array(z_array), (n,n))      # Returns list reshaped as the nxn cofactor matrix
#-----------------------------------------------------------------------------
def Det(z, n):
    """ Calculates the determinant of an input matrix 'z' with dimension 'n' by
    recursion, simplifying the problem to calculate the determinant of a 2x2 matrix"""

    # Basis of 2x2 in which all the determinant calculations can be simplified to use
    if n == 2:
        return    ( z[0][0] * z[1][1] ) - ( z[0][1] * z[1][0] )

    # If x>2 then continue to reduce down the determinant to be calculated till it is a simple 2x2 calculation as above.
    else:
        det = 0                                     # Set determinant equal to zero
        for j in range(n):                          # Looping through the first row only
            z_1 = np.delete(z, 0, 0)                # Delete the first row
            z_2 = np.delete(z_1, j, 1)              # Delete corresponding column
            z_3 = z[0][j] * Det(z_2, len(z_2))      # Calculate the determinant via recursion (reducing the problem)
            det += (-1)**(j) * z_3                  # Applying the correct signs for the determinant calculation, and add value for each index to list.

        return det                                  # Returns determinant of nxn matrix
#-----------------------------------------------------------------------------
def Inverse(z, n):
    """ Calculates the inverse of an 'nxn' matrix 'z'. Also indicates if the
    matrix inputted, 'z' does not have an inverse. """
    r = Cofactors(z, n)                             # Calculates Cofactor matrix
    d = Det(z, n)                                   # Calculates determinant
    if d == 0:                                      # Conditional statement for determinant equal to zero
        return ("This matrix has no inverse.")
    else:
        return ( 1 / d ) * np.transpose(r)          # Returns inverted matrix
#-----------------------------------------------------------------------------
def Time(function, *args):
    """ Times calculation time of a function. """
    start = timeit.default_timer()
    function(*args)
    stop = timeit.default_timer()
    return (stop - start)
#-----------------------------------------------------------------------------
def LUD(z, y):
    LUD1 = sc.linalg.lu_factor(z)
    LUD2 = sc.linalg.lu_solve(LUD1, y)
    return LUD2
#-----------------------------------------------------------------------------
def SVD(z, y):
    n = len(z)
    SVD = sc.linalg.svd(z)
    Sig  = sc.linalg.diagsvd(SVD[1], n,n)
    xbar = np.transpose(SVD[2]).dot(np.transpose(np.linalg.inv(Sig)).dot((np.transpose(SVD[0])).dot(y)))
    return xbar
#-----------------------------------------------------------------------------
def Analy(z, y):
    n = len(z)
    I = Inverse(z,n)
    return np.dot(I,y)
#-----------------------------------------------------------------------------
def timing_L_S(x, N_max):
    time_LUD =[]
    time_SVD = []

    size_n = range(2,N_max,1)

    for n in size_n:
        time_LUD1 =[]
        time_SVD1 = []
        z = np.random.randint(10, size=(n, n))
        y = np.random.randint(10, size = (n))
        for o in range(x):
            time_LUD1.append(Time(LUD, z, y))
            time_SVD1.append(Time(SVD, z, y))
        time_LUD.append(np.mean(time_LUD1))
        time_SVD.append(np.mean(time_SVD1))
    return time_LUD, time_SVD, size_n
#-----------------------------------------------------------------------------
def timing_A (x, N):
    time_Analy =[]
    size_n = range(2,N,1)
    for n in size_n:
        time_Analy1 = []
        z = np.random.randint(10, size=(n, n))
        y = np.random.randint(10, size = (n))
        for o in range(x):
            time_Analy1.append(Time(Analy, z, y))
        time_Analy.append(np.mean(time_Analy1))
    return time_Analy
#-----------------------------------------------------------------------------
def relative(x,y):
    x1, x2, x3, y1, y2, y3, z = (0, 90*(np.sqrt(3)), 45*(np.sqrt(3)), 0, 0, 135, 7)
    x1_rel = x1 - x
    x2_rel = x2 - x
    x3_rel = x3 - x
    y1_rel = y1 - y
    y2_rel = y2 - y
    y3_rel = y3 - y

    return [[x1_rel, x2_rel, x3_rel], [y1_rel, y2_rel, y3_rel], [z,z,z]]
#-----------------------------------------------------------------------------
def Matrix(x,y):
    matrix = np.zeros((3,3))
    position = relative(x,y)

    for i in range(3):
        matrix[0][i] = position[0][i] / (np.sqrt( (position[0][i])**2 + (position[1][i])**2 ) )
        matrix[1][i] = position[1][i] / (np.sqrt( (position[0][i])**2 + (position[1][i])**2 ) )
        matrix[2][i] = position[2][i] / (np.sqrt( (position[0][i])**2 + (position[1][i])**2 + (position[2][i])**2 ) )

    return matrix
# MENU -----------------------------------------------------------------------
Myinput = '0'
while Myinput != 'q':
    Myinput = input('Enter a choice,"1", "2", "3" or "q" to quit: ')
    print ("You entered the choice: ",Myinput)
#-----------------------------------------------------------------------------
# PART ONE -------------------------------------------------------------------
    if Myinput == '1':
        # INTRODUCTION -------------------------------------------------------
        print('-'*60)
        print("You have chosen part (1)")
        print("MATRIX INVERSION FOR LINEAR ALEGRBRA")
        print('-'*60)
        print("In this part of the exercise, square NxN matrices are inverted.")

        # INPUT AND CALCULATION ----------------------------------------------
        # Takes input from user for dimension of square matrix
        N = int(input("Please enter an integer value of the matrix dimension N: "))

        # Generates a random NxN matrix  with values between 0 and 9
        z = np.random.randint(10, size=(N, N))

        # Prints generated matrix and inverts (if possible)
        print("The following matrix has been generated;")
        print(z)
        print('-'*60)
        print("The inverted matrix is as follows;")
        print(Inverse(z,N))
        print('-'*60)
        #print(np.linalg.inv(z))

        # INVESTIGATION INTO SPEED -------------------------------------------
        print('-'*60)
        print("In this section, we will investigate how the dimension of a square nxn matrix \
influences the calculation time." )
        time = []
        size1 = range(2,10,1)
        for n in size1:
            gen = np.random.randint(10, size=(n, n))
            t = Time(Inverse, gen, n)
            time.append(t)
        # PLOTTING DETAILS  --------------------------------------------------
        plt.plot(size1, time )
        plt.title("Investigating how calculation time scales with dimension of the square matrix.")
        plt.ylabel("Calculation time (s)")
        plt.xlabel("Dimension of square matrix")
        plt.show()
        print('-'*60)
# END OF PART 1 --------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# PART TWO -------------------------------------------------------------------
    if Myinput == '2':
        # INTRODUCTION -------------------------------------------------------
        Input = 'input'
        while Input == 'input':
            Part = input("Please enter 'a' or 'b' for the following parts; \n PART A: Investigating \
general calculation times of the LUD and SVD methods for a matrix of N rows. \n PART B: Investigating \
singular matrices. \n >>> ")
            if Part == 'a' or Part == 'b':
                break

        if Part == 'a':
            x = int(input("Please enter a value for the number of repeats \n >>> "))
            N_max = int(input("Please enter the maximum number of rows for the matrix \n >>> "))

            time_LUD, time_SVD, size_n = timing_L_S(x, N_max)

            plt.title("Comparing calculation timing of the LUD and SVD methods.")
            plt.plot(size_n, time_LUD,  'g', label = "LUD")
            plt.plot(size_n, time_SVD, 'm', label = "SVD")
            plt.ylabel("Calculation time (s)")
            plt.xlabel("Dimension of square matrix")
            plt.legend()
            plt.show()

            extra = input("Would you like to view the LUD, the SVD and the analytical method for \
altogether for comparison for small matrices? \n If so, press y \n >>> ")
            if extra == 'y':
                time_Analy = timing_A(5,8)
                time_LUD1, time_SVD1, size_n1 = timing_L_S(5, 8)
                plt.title("Comparing calculation timing of the LUD, SVD and Analytical methods.")
                plt.plot(size_n1, time_LUD1,  'g', label = "LUD")
                plt.plot(size_n1, time_SVD1, 'm', label = "SVD")
                plt.plot(size_n1, time_Analy, 'c', label = "Analytical")
                plt.ylabel("Calculation time (s)")
                plt.xlabel("Dimension of square matrix")
                plt.legend()
                plt.show()
#-----------------------------------------------------------------------------
        if Part == 'b':
            print("In this section, the behaviour of the following set of equations is investigated \
when close to singular.")
            print(" x + y + z = 5 \n x + 2y - z = 10 \n 2x + 3y + kz = 15 \n where k is very small.")
            y = ([[5], [10], [15]])
            Input1 = 'input1'
            while Input1 == 'input1':
                subpart = input("Please select one of the following parts by entering 1, 2 or 3;\
\n PART i: Speed investigation \n PART ii: Accuracy of the LUD, SVD and analytical method \n \
PART iii: Testing how robust the methods are. ")
                if subpart == '1' or subpart == '2' or subpart == '3':
                    break

            if subpart == '1':
                x = 10
                time_LUD =[]
                time_SVD = []
                time_Analy = []
                a = []
                b = []
                c = []
                l = np.linspace(0.001, 0.1, 1000)
                comp1 =[]

                for k in l:
                    z = ([[1,1,1], [1,2,-1], [2,3,k]])
                    for j in range(x):
                        a.append(Time(LUD, z, y))
                        b.append(Time(SVD, z, y))
                        c.append(Time(Analy, z, y))
                    time_LUD.append(np.mean(a))
                    time_SVD.append(np.mean(b))
                    time_Analy.append(np.mean(c))
                plt.title("Comparing calculation timing of the LUD, SVD and Analytical methods.")
                plt.plot(l, time_LUD,  'g', label = "LUD")
                plt.plot(l, time_SVD, 'm', label = "SVD")
                plt.plot(l, time_Analy, 'c', label = "Analytical")
                plt.ylabel("Calculation time (s)")
                plt.xlabel("Dimension of square matrix")
                plt.legend()
                plt.show()

            if subpart == '2':

                LUD_accuracy =[]
                SVD_accuracy = []
                Analy_accuracy =[]
                a = []
                b = []
                c = []
                l = np.linspace(0.001, 0.1, 1000)
                comp1 =[]
                for k in l:
                    z = ([[1,1,1], [1,2,-1], [2,3,k]])
                    a.append(LUD(z,y))
                    b.append(SVD(z,y))
                    c.append(Analy(z,y))

                for i in a:
                    j = 15 - ( (2* i[0]) + (3 * i[1]) + (k * i[2]))
                    LUD_accuracy.append(np.abs(j))
                for i in b:
                    j = 15 - ( (2* i[0]) + (3 * i[1]) + (k * i[2]))
                    SVD_accuracy.append(np.abs(j))
                for i in c:
                    j = 15 - ( (2* i[0]) + (3 * i[1]) + (k * i[2]))
                    Analy_accuracy.append(np.abs(j))



                plt.plot(l, LUD_accuracy, 'm', label="LUD")
                plt.xlabel("Value of k")
                plt.ylabel("Absolute Error")
                plt.legend()
                plt.show()
                plt.plot(l, SVD_accuracy, 'g', label="SVD")
                plt.xlabel("Value of k")
                plt.ylabel("Absolute Error")
                plt.legend()
                plt.show()
                plt.plot(l, Analy_accuracy, 'r', label="Analytical")
                plt.xlabel("Value of k")
                plt.ylabel("Absolute Error")
                plt.legend()
                plt.show()


                f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
                ax1.plot(l, LUD_accuracy, 'm', label="LUD")
                ax1.legend()
                ax2.plot(l, SVD_accuracy, 'g', label="SVD")
                ax2.legend()
                ax3.plot(l, Analy_accuracy, 'r', label="Analytical")
                ax3.legend()
                ax2.set_xlabel("Value of k")
                ax1.set_ylabel("Absolute Error")
                plt.tight_layout()
                plt.show()

            if subpart == '3':
                print()
# END OF PART 2 --------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# PART THREE -----------------------------------------------------------------
    if Myinput == '3':
        # INTRODUCTION -------------------------------------------------------
            g = 9.81
            c = [[0],[0],[50*g]]
            x_lin = np.linspace(1e-6, 45*(np.sqrt(3)), 5)
            y_lin = np.linspace(1e-6, 45, 5)
            index = 0
            T = np.zeros((len(x_lin), len(y_lin)))

            for i, x in enumerate(x_lin):
                for j, y in enumerate(y_lin):
                    print(Matrix(x,y))
                    print(Matrix(x,y).dot(c))
                    T[i][j] = (sc.linalg.inv(Matrix(x,y))).dot(c)[index]



            plt.imshow(T)
            plt.show()
            """fig = plt.figure()                          # 3D figure plotting
            ax = Axes3D(fig)
            ax.plot(x_lin, y_lin, T1, 'r')
            plt.show()"""











print ("You have chosen to quit.\nGoodbye.")
