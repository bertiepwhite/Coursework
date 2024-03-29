# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
from cython.view cimport array as cvarray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libc.math cimport sqrt
import time
cimport openmp

def main():
    cdef:
        int N = 125              # Number of particles
        int N_t = 2              # Number of time steps
        int N_l = 5             # Cube of side N_l makes N particles
        double G = 6.67e-11
        double M_solar = 2e30
        double M_Gal = 1e12 * M_solar
        double radius = 1.234271032e21
        double l = radius/N_l
        double dt = 365.25*24*60*60*250000
        int i,j,k,index
        double GmM_rrr
        double h

    start = openmp.omp_get_wtime()

    position   = np.zeros(shape=(N,3),dtype=np.float_)
    velocities = np.zeros(shape=(N,3),dtype=np.float_)
    forces     = np.zeros(shape=(N,3), dtype=np.float_)
    masses     = np.ones(N, dtype=np.float_) * M_Gal/512
    pos_r      = np.zeros(3, dtype=np.float_) # Position, relative


    for i in range(N_l):
        for j in range(N_l):
            for k in range(N_l):
                index = (i*N_l**2+j*N_l+k)
                position[index][0] = i*l-radius/2 + 0.01 * l
                position[index][1] = j*l-radius/2
                position[index][2] = k*l-radius/2
                h = position[index][0]**2 + position[index][1]**2 + position[index][2]**2
                velocities[index][0] = -((j*l-radius/2)/sqrt(h))*2000
                velocities[index][1] =  ((j*l-radius/2)/sqrt(h))*2000

    cdef:
        double[:,:] pos_view = position
        double[:,:] vel_view = velocities
        double[:,:] frc_view = forces
        double[:]   mas_view = masses
        double[:]   pos_r_v  = pos_r

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(N):
        for j in range(i,N):
            for k in range(3):
                pos_r_v[k] = pos_view[i][k] - pos_view[j][k]
            h = pos_r_v[0]**2 + pos_r_v[1]**2 + pos_r_v[2]**2+(0.05*l)**2
            GmM_rrr = G*mas_view[i]*mas_view[j]/(h*sqrt(h))
            for k in range(3):
                frc_view[i][k] -= GmM_rrr*pos_r_v[k]
                frc_view[j][k] += GmM_rrr*pos_r_v[k]

    for i in range(N):
        for k in range(3):
            vel_view[i][k] += (frc_view[i][k]*dt/mas_view[i])*0.5

    for time in range(N_t):
        print("Iteration: ",time,"/",N_t)

        for i in range(N):
            frc_view[i][0] = 0.0; frc_view[i][1] = 0.0; frc_view[i][2] = 0.0;

        #print("Before: ", frc_view[100][1] )
        for i in range(N):
            for j in range(i+1,N):
                for k in range(3):
                    pos_r_v[k] = pos_view[i][k] - pos_view[j][k]

                h = pos_r_v[0]**2 + pos_r_v[1]**2 + pos_r_v[2]**2

                GmM_rrr = G*mas_view[i]*mas_view[j]/(h*sqrt(h))

                for k in range(3):
                    frc_view[i][k] -= GmM_rrr*pos_r_v[k]
                    frc_view[j][k] += GmM_rrr*pos_r_v[k]
        #print("After: ", frc_view[100][1] )
        # print("Before: ", pos_view[100][1], vel_view[100][1], frc_view[100][1] )
        for i in range(N):
            for k in range(3):
                pos_view[i][k] += (vel_view[i][k]*dt)
                vel_view[i][k] += (frc_view[i][k]*dt/mas_view[i])
        if time == 0:
            print(forces)

    final = openmp.omp_get_wtime()
    diff = final-start
    print(diff)
        # print("After: ", pos_view[100][1], vel_view[100][1], frc_view[100][1] )
    ax.set_xlim3d(left = -radius/2, right=radius/2)
    ax.set_ylim3d(bottom = -radius/2, top=radius/2)
    ax.set_zlim3d(bottom = -radius/2, top=radius/2)
    ax.scatter(pos_view[:,0],pos_view[:,1],pos_view[:,2], color = 'r',s=1)
    plt.show()
