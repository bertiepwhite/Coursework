import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


f = pd.read_csv("Gravity_output_cl.csv")
"""
x = np.array(f['x'],dtype=float)
y = np.array(f['y'],dtype=float)
z = np.array(f['z'],dtype=float)
"""
"""fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x,y,z,s=1)"""
"""plt.scatter(x,y)"""
E = np.array(f['E'],dtype=float)
AU = 1
plt.plot(E)
#plt.ylim((0,-5000))
plt.show()
"""plt.xlim((-90*AU,90*AU))
plt.ylim((-90*AU,90*AU))
plt.show()"""

#Energies = np.genfromtxt("Gravity_energies.csv")
#plt.plot(Energies)
#plt.show()
