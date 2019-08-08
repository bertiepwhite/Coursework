import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 22}

matplotlib.rc('font', **font)

def func(x,m,c):
    return m*x+c
"""fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x,y,z,s=1)"""
def spatial_plot():

    f = pd.read_csv("Gravity_output.csv")
    x = np.array(f['x'],dtype=float)
    y = np.array(f['y'],dtype=float)
    z = np.array(f['z'],dtype=float)
    plt.scatter(x,y)

    AU = 1

    plt.xlim((-90*AU,90*AU))
    plt.ylim((-90*AU,90*AU))
    plt.show()

def Energy_plot():
    EnergiesPar = np.genfromtxt("energies.csv")
    #EnergiesLin = np.genfromtxt("Gravity_energies_lin.csv")
    #EnergiesGPU = np.genfromtxt("Gravity_energies_GPU.csv")

    x = np.arange(200000)
    x = x*1/(24/60*6)

    Energies_smoothed = np.zeros(20)
    Energies_smoothed_x = np.zeros(20)
    #for i in range(20):
    #    Energies_smoothed[i] = np.mean(EnergiesPar[i*10000:(i+1)*10000])
    #    Energies_smoothed_x[i] = x[int((i+0.5)*10000)]




    plt.plot(EnergiesPar,linewidth=4,color = 'r',label="Par")
    #plt.plot(x,EnergiesLin,linestyle='--',linewidth=4,dashes=(10, 10),color ='b',label="Lin")
    #plt.plot(x,EnergiesGPU[:-1:],linewidth=4,color ='g',label="GPU")
    #plt.plot(Energies_smoothed_x,Energies_smoothed,color='k',linewidth=4,label="Ave")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.show()

def Energy_dt_plot():
    steps = [100000,200000,300000,400000,500000,600000,700000,800000,900000]
    steps = np.array(steps)
    steps = np.sort(steps)
    std = np.zeros_like(steps)
    dt = np.zeros_like(steps)
    for i,step in enumerate(steps):
        print(i)
        file = str(step) + "energies.csv"
        file_array = np.genfromtxt(file)
        time = np.arange(file_array.shape[0])
        std[i] = np.std(file_array)
        dt[i] = (1000000000000*1/(24.0*60.0*60.0))/step
        time = time*dt[i]
        sqr=(file_array[:-1:]-file_array[1::])**2
        sum_root = np.sum(sqr)**0.5
        std[i] = sum_root
        #plt.plot(time,file_array,label=str(dt[i]))
    #plt.legend()
    #plt.show()
    popt,pcov = curve_fit(func,np.log(dt[:5]),np.log(std[:5]))
    plt.plot(np.log(dt[:5]),func(np.log(dt[:5]),popt[0],popt[1]),color = 'k')
    plt.scatter(np.log(dt),np.log(std),color = 'r')
    plt.xlabel("Log(dt)")
    plt.ylabel("Log(error)")
    plt.show()
