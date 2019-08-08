import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

FullPositions = np.load('FullPositions.npy')
#FullPositions = np.genfromtxt("Gravity_output_full.csv")
#FullPositions = FullPositions.reshape((128,13824/4,3))
vid_name = 'Gravity.gif'

fig = plt.figure()
ax = fig.gca(projection='3d')
radius = 1.234271032e21
frames = FullPositions.shape[0]


def update_img(n):
    print(n)
    plt.cla()
    slice = FullPositions[n]
    ax.set_xlim3d(-radius,radius)
    ax.set_ylim3d(-radius,radius)
    ax.set_zlim3d(-radius,radius)
    scatter = ax.scatter(slice[:,0],slice[:,1],slice[:,2])
    return scatter


ani = animation.FuncAnimation(fig, update_img, frames-1)
writer = animation.writers['pillow'](fps = 15)
ani.save(vid_name, writer = writer, dpi = 120)
ani._stop()
