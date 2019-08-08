import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

#FullPositions = np.load('FullPositions.npy')
FullPositions = np.genfromtxt("Gravity_output_full.csv")
FullPositions = FullPositions.reshape((-1,50,3))
vid_name = 'Gravity.gif'

frames = int(FullPositions.shape[0])
AU = 1 #149597870700

def video_make():

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    def update_img(n):
        print(n)
        plt.cla()
        slice = FullPositions[n]
        ax.set_xlim3d(-50*AU,50*AU)
        ax.set_ylim3d(-50*AU,50*AU)
        ax.set_zlim3d(-50*AU,50*AU)
        scatter = ax.scatter(slice[:,0],slice[:,1],slice[:,2])
        return scatter


    ani = animation.FuncAnimation(fig, update_img, frames-1)
    writer = animation.writers['pillow'](fps=20)
    ani.save(vid_name, writer = writer, dpi = 120)
    ani._stop()

def plot_trajectory(N):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig1 = plt.figure()
    ax1 = fig1.gca(projection=None)
    fig2 = plt.figure()
    ax2 = fig2.gca(projection=None)

    for i in range(N):
        pos = FullPositions[:,i]
        ax.plot(pos[:,0],pos[:,1],pos[:,2])
        ax1.plot(pos[:,0],pos[:,1])
        ax1.set_xlim(-50*AU,50*AU)
        ax1.set_ylim(-50*AU,50*AU)
        ax2.plot(pos[:,1],pos[:,2])
        ax2.set_xlim(-50*AU,50*AU)
        ax2.set_ylim(-10*AU,10*AU)
    for i in range(9,N):
        pos = FullPositions[:,i][:2:]
        ax.plot(pos[:,0],pos[:,1],pos[:,2])
        ax1.plot(pos[:,0],pos[:,1],linewidth=0.25,color='k')
        ax1.set_xlim(-50*AU,50*AU)
        ax1.set_ylim(-50*AU,50*AU)
        ax2.plot(pos[:,1],pos[:,2],linewidth=0.25,color='k')
        ax2.set_xlim(-50*AU,50*AU)
        ax2.set_ylim(-10*AU,10*AU)
    plt.show()
