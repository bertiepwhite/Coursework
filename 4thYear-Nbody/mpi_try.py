from mpi4py import MPI
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def init_arrays(N,l,radius):
    positions   = zeros(shape=(N,3))
    velocities  = zeros(shape=(N,3))

    lattice_side = ceil(power(N,1/3))
    i_pos,j_pos,k_pos = 0,0,0

    for i in range(N):
        positions[i,0] = i_pos*l-radius/2 + 0.01 * l #Offsetting so no particles are on the origin
        positions[i,1] = j_pos*l-radius/2
        positions[i,2] = (k_pos*l-radius/2)

        h = sqrt(sum(power(positions[i],2))) # Try to numpy as much as possible
                                             # since this is basically fortran

        velocities[i,1] =  ((i_pos*l-radius/2)/h)*6000
        velocities[i,0] = -((j_pos*l-radius/2)/h)*6000
        velocities[i,2] = 0

        i_pos += 1
        if i_pos == lattice_side:
            i_pos  = 0
            j_pos += 1
        if j_pos == lattice_side:
            j_pos  = 0
            k_pos += 1

    forces = zeros(shape=(N,3))
    return positions,velocities,forces

# Try couple ways of this
def force_calc(positions,forces,offset,chunk,G,M,N):

    positions2 = positions.copy()
    positions2 = roll(positions2,1,axis=0)
    for i in range(N-1):
        rel_pos = positions[offset:(offset+chunk)]-positions2[offset:(offset+chunk)]
        rel_pos_mag = sum(power(rel_pos,2),axis=1)
        forces[offset:(offset+chunk)][:,0] -= multiply(
                                                G*M*M/(power(rel_pos_mag,(3/2))),
                                                rel_pos[:,0])
        forces[offset:(offset+chunk)][:,1] -= multiply(
                                                G*M*M/(power(rel_pos_mag,(3/2))),
                                                rel_pos[:,1])
        forces[offset:(offset+chunk)][:,2] -= multiply(
                                                G*M*M/(power(rel_pos_mag,(3/2))),
                                                rel_pos[:,2])
        positions2 = roll(positions2,1,axis=0)
    return forces
    """
    for i in range(offset,offset+chunk):
        for j in range(N):
            rel_pos = positions[i]-positions[j]
            h = rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2
            GMM_rrr = G*M*M/(h**(3/2))
            forces[i] -= GMM_rrr * rel_pos
    return forces
    """
# MPI variables and stuff
comm = MPI.COMM_WORLD

numranks = comm.Get_size()
rank = comm.Get_rank()

Master = 0
Master_tag = 1
Worker_tag = 2

# Normal python variables and stuff
N = 4096
N_t = 128
G = 6.67e-11
M_solar = 2e30
M_Gal = 1e12 * M_solar
M = M_Gal/N
radius = 1.234271032e21
l = radius/(N**(1/3))
dt = 365.25*24*60*60*500000

rank = comm.Get_rank()

chunk = 2*N // (2*numranks+1)

# Simply Initialising
if rank == Master:
    # Initialising arrays
    start = MPI.Wtime()
    positions,velocities,forces = init_arrays(N,l,radius)
    PositionsFull = zeros(shape=(N_t,N,3))
if rank != Master:
    positions = zeros(shape=(N,3))
    forces    = zeros(shape=(N,3))

for t in range(N_t):

    if rank == Master:
        offset = 0
        for worker_num in range(1,numranks):
            comm.send(offset,dest=worker_num,tag=Master_tag)
            comm.Send(positions,dest=worker_num,tag=Master_tag)
            offset += chunk

        forces[::] = 0
        forces = force_calc(positions,forces,offset,N-offset,G,M,N)

        for worker_num in range(1,numranks):
            offset = comm.recv(source = worker_num, tag=Worker_tag)
            comm.Recv([forces[offset:offset+chunk],N*3,MPI.DOUBLE],source=worker_num,tag=Worker_tag)

        positions  += velocities*dt
        velocities += forces*dt/M
        PositionsFull[t] = positions

    if rank != Master:

        forces[::] = 0

        offset = comm.recv(source=Master, tag=Master_tag)
        if t == 0: print(offset)
        comm.Recv([positions,N*3,MPI.DOUBLE],source=Master,tag=Master_tag)

        forces = force_calc(positions,forces,offset,chunk,G,M,N)

        comm.send(offset,dest=Master,tag=Worker_tag)
        comm.Send(forces[offset:offset+chunk],dest=Master,tag=Worker_tag)
if rank == Master:
    end = MPI.Wtime()
    save("FullPositions.npy",PositionsFull)
    print("Elapsed time = %lf" % (end-start))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(left = -radius/2, right=radius/2)
    ax.set_ylim3d(bottom = -radius/2, top=radius/2)
    ax.set_zlim3d(bottom = -radius/2, top=radius/2)
    ax.scatter(positions[:,0],positions[:,1],positions[:,2], color = 'r',s=2)
    plt.show()
