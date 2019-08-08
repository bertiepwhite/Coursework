import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
import math as mt
import time


from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from random import randint
from scipy.integrate import simps
from scipy.stats import chisquare


"""-------------------------------Classes-----------------------------------"""

class Field():
    """
    Field has been kept general and not specified down to a capacitor to allow
    for the method of images checks, and so there is no needless extra specifying.
    A number of Capacitor centric functions have been created below.

    Guass-striedel method was broken into 2 functions (one_iterate_gauss and
    grid_iterate_gauss) to aid with bug fixing, nothing is gained by forcing them
    into one. Jacobi was simple enough to fix without so has been left as one.

    Throughout many issues were had with x and y axis being flipped. The final
    solution found involves sometimes x and y are assigned and used in an unusual
    manner. However this final system produces a reliable result in the way
    expected.
    """
    def __init__(self, x_y, gauss_jacobi_jacobi_a = 'jacobi_a'):
        #x_y should be inputted as a tuple in the form of (x,y), the number of
        # nodes are set and then an array of that size is made.
        self.nodes = 10000
        self.set_nodes(x_y)
        self.scalar_field = np.zeros((self.y_nodes,self.x_nodes))

        self.count = 0
        self.converged = False
        self.method = gauss_jacobi_jacobi_a

        #Boundaries are stored as a list of tuples, and the ref_col and ref_row
        # are the row and columns used to check for convergence
        self.boundaries = []
        self.boundaries_alt = []
        self.ref_col = int(self.x_nodes/2)
        self.ref_row = int(self.y_nodes/2)

    def reset(self):
        #Resets the relevant values to allow manual changing of other varaibles
        self.scalar_field = np.zeros((self.y_nodes, self.x_nodes))
        self.count = 0
        self.converged = False
        self.ref_col = int(self.x_nodes/2)
        self.ref_row = int(self.y_nodes/2)

    def set_nodes(self, x_y):
        #Unpacks the tuple then scales the sides so roughly self.nodes are
        # contained in the array.
        x,y = x_y
        self.x_nodes = int(np.sqrt(self.nodes/(x*y))*x)
        self.y_nodes = int(np.sqrt(self.nodes/(x*y))*y)

    def make_node_array(self):
        #Creates an array with 4 in the centre 3 on the edge and 2 in the corners
        self.node_array = np.zeros_like(self.scalar_field)
        #Set all values to 4
        self.node_array += 4
        #Removing 1 from each edge, the corners get caught twice so go to the 2
        #required
        self.node_array[0] -= 1
        self.node_array[-1] -= 1
        self.node_array[:,0] -= 1
        self.node_array[:,-1] -= 1

    def set_boundary(self, value, start, end, first_set = True):
        #The boundary start and end are inclusive first_set turns of the saving
        #of the conditions if they've already been saved and are just being
        #reapplied.

        #If the start and end are on the same row extract that row and slice
        # between those 2 points. Sets that to the value. Also adds those points
        # to the stored boundaries.
        if start[0] == end[0]:
            self.scalar_field[start[0]][start[1]:end[1]+1] = value
            point_list = [(start[0],i) for i in range(start[1],(end[1]+1))]
            if first_set:
                self.boundaries += point_list

        #Does the same but for the column, extracting the column than slicing
        # that column. Adding the boundary values to the boundary arrays.
        if start[1] == end[1]:
            self.scalar_field[:,start[1]][start[0]:(end[0]+1)] = value
            point_list = [(i,start[1]) for i in range(start[0],(end[0]+1))]
            if first_set:
                self.boundaries += point_list
        if first_set:
            self.boundaries_alt .append((value,start,end))

    def make_iterator(self):
        #This is called several times always with the same arugements, so easier
        # to change and test if stored in a function. 'K' gies the most effecient
        #order
        self.iterator = np.nditer(self.scalar_field, flags = ['multi_index'],
                                  op_flags = ['readwrite'], order = 'K')

    def one_iterate_gauss(self):
        #Processing a single point with the gauss regime.

        #Making an iterator object if one does not already exist.
        if not 'iterator' in self.__dict__:
            self.make_iterator()

        #One of the x-y being odd fixes. Reversing the multi index and checking
        # if that is in the boundaries list. If it is going to the next iteration
        # and ending.
        if self.iterator.multi_index in self.boundaries:
            self.iterator.iternext()
            return

        #Sums all the elements around it if it can and keeps track how many it
        # has summed over
        i,j = self.iterator.multi_index
        count = 0
        summation = 0
        if i != 0:
            summation += self.scalar_field[i-1][j]
            count += 1
        if i != (self.y_nodes - 1):
            summation += self.scalar_field[i+1][j]
            count += 1
        if j != 0:
            summation += self.scalar_field[i][j-1]
            count += 1
        if j != (self.x_nodes - 1):
            summation += self.scalar_field[i][j+1]
            count += 1

        #Finding the average of the sums
        value = summation/count
        #Loses resolution but avoids floating point erros from compounding
        if np.abs(value) < 1e-14:
            value = 0

        #Sets current array position to the average
        self.iterator[0] = value

        #Moves to next iterations
        self.iterator.iternext()

    def grid_iterate_gauss(self):
        #Makes the iterator and then iterates everypoint on the array
        if not 'iterator' in self.__dict__:
            self.make_iterator()

        #Loops through the entire array iterating each point
        while not self.iterator.finished:
            self.one_iterate_gauss()

        self.make_iterator()
        self.count += 1

    def grid_iterate_jacobi(self):
        #Iterates the whole grid at once

        #Uses np.nditer to more effeciently iterate over the whole array
        self.make_iterator()
        new_jacobi = np.zeros((self.y_nodes, self.x_nodes))

        #Conitnues looping until the whole grid has been iterated over
        while not self.iterator.finished:

            i,j = self.iterator.multi_index

            #If the
            if self.iterator.multi_index in self.boundaries:
                new_jacobi[i][j] = self.iterator[0]
                self.iterator.iternext()
                continue

            count = 0
            summation = 0

            #Adds all the points around it which are on teh grid, coutning how
            #Many it sums over over
            if i != 0:
                summation += self.scalar_field[i-1][j]
                count += 1
            if j != 0:
                summation += self.scalar_field[i][j-1]
                count += 1
            if i != (self.y_nodes-1):
                summation += self.scalar_field[i+1][j]
                count += 1
            if j != (self.x_nodes-1):
                summation += self.scalar_field[i][j+1]
                count += 1

            #Averages the points summed
            value = summation/count
            #Removes floating point errors that could easily compound, However
            # this is a slight loss in resolution
            if np.abs(value) < 1e-14:
                value = 0

            #Updates the value in the new grid
            new_jacobi[i][j] = value

            self.iterator.iternext()

        #Sets old grid equal to new grid
        self.scalar_field = new_jacobi.copy()

        self.count += 1

    def grid_iterate_jacobi_alt(self):
        #Sums the entiere grid at once, translates the grid to the left and
        #adds the translation to an empty grid, then translates the grid to the
        #right adding the values to the previoussly saved ones, then does this
        #again for up and down, finally dividing through by how many points have
        #been summed to any grid

        #The new grid
        new_jacobi = np.zeros_like(self.scalar_field)

        #Takes the all but the first value of the old grid and sums it in the
        #place of all but the last value of the new grid.
        new_jacobi[:-1] += self.scalar_field[1:].copy()
        #Does reverse of the previous
        new_jacobi[1:] += self.scalar_field[:-1].copy()
        #Does the process for up and down
        new_jacobi[:,1:] += self.scalar_field[:,:-1].copy()
        new_jacobi[:,:-1] += self.scalar_field[:,1:].copy()

        self.scalar_field = new_jacobi/self.node_array

        #Boundary conditions are lost in this process so have to be reinstated
        for value,start,end in self.boundaries_alt:
            #First_set is used to avoiding adding the boundary conditions to the
            #list repeatdly
            self.set_boundary(value,start,end,first_set=False)

        self.count += 1

    def grid_iterate_gauss_alt(self, print_tests = False):
        #This is tough to follow if my jacobi explanation wasnt good enough, but
        # here goes. Will be using an anology to a chess board a lot.

        #Setting everyother value to 0, calling them the black squares
        self.scalar_field[::2][:,::2] = 0
        self.scalar_field[1::2][:,1::2] = 0

        #Saving an intermediate array to avoid over writing over values still
        #required
        new = self.scalar_field.copy()

        #Translating the black squares up down left and right like in the jacobi
        #method to calculate white squares, which will all be new values.
        new[:-1] += self.scalar_field[1:].copy()
        new[1:] += self.scalar_field[:-1].copy()
        new[:,1:] += self.scalar_field[:,:-1].copy()
        new[:,:-1] += self.scalar_field[:,1:].copy()

        self.scalar_field = new.copy()

        #Scaling everything down to whta it should be
        self.scalar_field = self.scalar_field/self.node_array

        #Reinstating boundary conditions
        for value,start,end in self.boundaries_alt:
            self.set_boundary(value,start,end,first_set=False)

        #Setting the black squares to 0 so they can be updated from the new
        #white square values.
        self.scalar_field[1::2][:,::2] = 0
        self.scalar_field[::2][:,1::2] = 0

        #Savin the temproary array with white sqaures filled in
        new = self.scalar_field.copy()

        #Adding to just the emtpy black sqaures in the new array
        new[:-1] += self.scalar_field[1:].copy()
        new[1:] += self.scalar_field[:-1].copy()
        new[:,1:] += self.scalar_field[:,:-1].copy()
        new[:,:-1] += self.scalar_field[:,1:].copy()

        #Updating the scalar array
        self.scalar_field = new.copy()

        #Scaling just the black squares since the white squars have already been scaled
        self.scalar_field[1::2][:,::2] = self.scalar_field[1::2][:,::2] / self.node_array[1::2][:,::2]
        self.scalar_field[::2][:,1::2] = self.scalar_field[::2][:,1::2] / self.node_array[::2][:,1::2]

        #Reseting wrecked boundary condiitons
        for value,start,end in self.boundaries_alt:
            self.set_boundary(value,start,end,first_set=False)

        self.count += 1
        #And if you followed that well done

    def convergence_check(self):
        #Should be applied after a grid iterater, takes a sample from a row
        #And a column and checks to see the change. A bit of an overuse of
        # copy() in here but has to be done to ensure no issues.

        #If no rows or cols are currently stored just stores the current ones
        # and stores the convergence
        if not 'converge_row' in self.__dict__:

            self.converge_row = self.scalar_field[self.ref_row].copy()
            self.converge_col = self.scalar_field[:,self.ref_col].copy()

            self.check_max = np.max(self.scalar_field)/1000

            print("Reference row and column saved")
            return

        #Stores current rows
        row_check = self.scalar_field[self.ref_row].copy()
        col_check = self.scalar_field[:,self.ref_col].copy()

        #Finds the absolute of the difference between stored and current
        check1 = np.abs(self.converge_row-row_check)
        check2 = np.abs(self.converge_col-col_check)

        self.check = sum(check1) + sum(check2)

        #Checks against convergence criteria
        if self.check < self.check_max:
            self.converged = True

        #Updates the stored row and col
        self.converge_row = row_check.copy()
        self.converge_col = col_check.copy()

    def iterate_to_converge(self):
        #Keeps on iterating via the chosen method until convergence. The code is
        # split into 4 looops to cut the number of if tests.

        #Run to create the first reference row and col
        self.convergence_check()

        if self.method == 'gauss':
            while not self.converged:
                self.grid_iterate_gauss()
                self.convergence_check()

        if self.method == 'jacobi':
            while not self.converged:
                self.grid_iterate_jacobi()
                self.convergence_check()

        if self.method == 'jacobi_a':
            self.make_node_array()
            while not self.converged:
                self.grid_iterate_jacobi_alt()
                self.convergence_check()

        if self.method == 'gauss_a':
            self.make_node_array()
            while not self.converged:
                self.grid_iterate_gauss_alt()
                self.convergence_check()

    def quick_plot(self):
        #Plots the scalar field for quick checking
        plt.imshow(self.scalar_field, cmap = 'inferno')
        plt.xlabel("X Nodes")
        plt.ylabel("Y Nodes")
        cb = plt.colorbar()
        cb.set_label("Potential (V)", rotation=270)
        plt.show()

    def E_field_get(self):
        #Out puts 2 arrays
        self.dy, self.dx = np.gradient(-self.scalar_field)

        #Works out the magnitude for each complimentary points in the array
        self.E_field = np.hypot(self.dx,self.dy)

    def plot_field(self, field_lines = True, field = True):
        #Plots the electric field

        #streamplot requires the axises so making them here
        x = np.arange(self.x_nodes)
        y = np.arange(self.y_nodes)

        #Making the E_field
        if 'E_field' not in self.__dict__:
            self.E_field_get()

        #Generates the fig and ax objects to add the streamplot if needed
        fig = plt.figure(11)
        ax = fig.gca()
        #Adds the field as a heat map if wanted
        if field:
            plt.imshow(self.E_field, origin = 'lower', extent = [0,self.x_nodes,0,self.y_nodes])
            cb = plt.colorbar()
            cb.set_label("Electric Field (Vm^-1)", rotation = 270, labelpad = 10)
        #Adds tehf field lines if requred, which is the negative of the
        #  differential of the differential
        if field_lines:
            ax.streamplot(x,y, -self.dx, -self.dy, color = 'r', density = 0.5)
            ax.set_xlim([0,self.x_nodes])
            ax.set_ylim([0,self.y_nodes])
        plt.show()

class Rod():
    """
    A class for modelling heat diffusion through a 1d rod. The rod can either
    have no heat lose to the environment or have the the end held at 0. The first
    node is the heat source outside the node, and the final node is the ice bath
    or external air.
    """

    def __init__(self, end_temp, time_step, nodes, ice = False):

        #The geomtry of the rod
        self.length = 0.5
        self.nodes = nodes
        self.x_step = self.length/self.nodes

        #The time based variables
        self.time = 0
        self.time_step = time_step

        #temperature variables
        main_temp = 20
        self.end_temp = end_temp
        if ice:
            self.ice = True
            self.ice_temp = 0
        else:
            self.ice = False

        #Setting the starting temperatures across the rod
        self.temp_array = np.array([20]*self.nodes)
        self.temp_array[0] = end_temp
        if ice:
            self.temp_array[-1] = 0

        #Allows for easy changing of the conductivity,density and capacity
        conductivity = 59
        density = 7900
        capacity = 450

        self.alpha = conductivity/(density*capacity)

        #Creating the matrix since only need the decomposition of this matrix
        # we do not make it a class variable
        self.make_iteration_matrix()

    def reset(self):
        #Resets all the values that change
        end_temp = self.temp_array[0]
        self.temp_array = np.array([20]*self.nodes)
        self.temp_array[0] = end_temp
        if self.ice:
            self.temp_array[-1] = 0

        self.time = 0

    def make_iteration_matrix(self, returns = False):
        #Makes the interation matrix starting with a zero everywhere matrix
        # and fillin the 3 relevant diagnols

        C = self.alpha*self.time_step/(self.x_step**2)

        matrix = np.zeros((self.nodes, self.nodes))
        #Filling in the middle diagonal
        np.fill_diagonal(matrix, round((1 + 2*C),1))

        #Filling in the bottom diagnol
        np.fill_diagonal(matrix[1:], round(-C,1))

        #Filling in the top diagnol
        np.fill_diagonal(matrix[:,1:], round(-C,1))

        #sets first and final rows to just be 1s in the corner
        matrix[0][0] = 1
        matrix[-1][-1] = 1+C
        matrix[0][1] = 0

        #Changes end values if ice is needed
        if self.ice:
            matrix[-1][-1] = 1
            matrix[-1][-2] = 0

        #Stores just the composition since thats what's required
        self.lu_piv = spla.lu_factor(matrix)

        #Allows the matrix to be retrieved
        if returns:
            return matrix

    def take_time_step(self):
        #Iiterates the whole array by solving the simultaenous equations

        #Solving the simultaenous equtions
        stepped_array = spla.lu_solve(self.lu_piv, self.temp_array)

        #Fixing the end temp
        stepped_array[0] = self.end_temp

        self.temp_array = stepped_array.copy()

        self.time += self.time_step

    def quick_plot(self):
        #Plots the temperature array
        x = [i*self.x_step for i in range(self.nodes)]
        plt.plot(x,self.temp_array)
        plt.xlabel("Position on rod")
        plt.ylabel("Temperature")
        plt.show()

    def full_plot(self, max_time):
        #Plots several nodes temperatures offset by their positions

        #If any time steps have been taking giving the user (aka me) to bail out
        # this shouldn't come up
        if self.time != 0:
            check = input("Iterations have been run, would you like to continue\
                           pless enter [y/n]: ")
            if check == 'n':
                return

        #Finding the number of time steps that will be taken
        y_points = mt.ceil(max_time/self.time_step)
        #Creating a 0 array to replace data with
        full_data = np.zeros((y_points, self.nodes))
        #Storing each nodes temperature
        while self.time < max_time:
            i = int(round(self.time/self.time_step,0))
            full_data[i] = self.temp_array
            self.take_time_step()

        #Reordering data so rows are the temperatures of points starting from
        #cold end and finishing at hot
        full_data = np.transpose(full_data)[::-1]

        x_points = np.linspace(0, self.time, i+1)

        #Extracting a sub section of the nodes for the plot points temperatures
        for i,row in enumerate(full_data[::10]):
            row += i*10
            plt.plot(x_points, row)
        plt.plot(x_points, full_data[-1])
        plt.xlabel("Time")
        plt.ylabel("Temperature + node number")
        plt.show()

    def step_plot(self, max_time, plot_interval):
        #Generates several plots of the tempearture positions for different times
        # to see the trend

        #Gives user option to skip if its already generated
        if self.time != 0:
            check = input("You are starting from an already iterated rod, would you like to continue [y/n]?")
            if check != 'y':
                print("You have not selected y exiting")
                return

        #Generates the x values for the plots
        x = np.linspace(0,0.5,num = 100)

        #Plots the initial condition
        plt.plot(x,self.temp_array, label = "Time = 0s")

        #takes the steps and if the time is a required time, its divisible by the
        #plot factor, plots that data
        while self.time < max_time:
            self.take_time_step()
            if self.time % plot_interval == 0:
                plt.plot(x,self.temp_array, label = ("Time = " + str(self.time) + "s"))

        #Usual plotting jargon
        plt.xlabel("Distance along the rod")
        plt.ylabel("Temperature")
        plt.legend( bbox_to_anchor=(1, 1))
        plt.show()

    def iterate_to_converge(self):
        #Keeps iterating until the convergence condition is met, if its in ice
        #tahts the sum change is equal to =< 1.
        old_array = self.temp_array.copy()

        #Takes time step and then sums the difference between the old and new
        # values breaking the loop when this is less than 1.
        if self.ice:
            check = 1000
            while check > 1:
                self.take_time_step()
                check = sum(np.abs(old_array-self.temp_array))
                old_array = self.temp_array.copy()

        #If all values are greater than 950 call it convrged
        if not self.ice:
            while np.mean(self.temp_array) < 950:
                self.take_time_step()

    def alpha_explore(self, returns = True):
        #See the relationship between alpha and the time for the rod to reach a
        #stable configuration

        #Using hti srange of alpha
        alphas = np.linspace(1e-5,1e-3,100)
        times = []
        for alpha in alphas:
            #Sets alpha and updates the iteration matrix
            self.alpha = alpha
            self.make_iteration_matrix()
            #Times the convergence
            self.iterate_to_converge()
            times.append(self.time)
            #Resets the rod
            self.reset()

        #Same old plotting jargon
        plt.plot(alphas,times)
        plt.xlim(xmin = 0)
        plt.ylim(ymin = 0)
        plt.xlabel("Alpha")
        plt.ylabel("Time to converge")
        plt.show()

        #Returns the list so can be exported to
        if returns:
            return alphas, times

class Rod_2():
    def __init__(self):
        #Setting this up is going to be difficult so will have minimal possible
        #things changable.

        #Geometery
        self.length = 0.5
        self.width = 0.1

        self.x_nodes = 50
        self.y_nodes = 25

        #Size of steps
        self.x_step = self.length/self.x_nodes
        self.y_step = self.length/self.y_nodes

        self.time_step = 50
        self.time = 0

        self.temp_field = np.array([[20]*self.x_nodes]*self.y_nodes)
        #Setting an array with a band of heating
        self.temp_field[:,0][int(self.y_nodes/4): int(self.y_nodes*(3/4))] = 1000

        #Based on steel
        conductivity = 59
        density = 7900
        capacity = 450

        #Set alpha based on those
        self.alpha = conductivity/(density*capacity)

        #Generates both the x and y matrices which will be different since the
        #x step and y step vary.
        self.make_matrices()

    def quick_plot(self):
        #Plotting the entire rod as a heat map
        plt.imshow(self.temp_field, extent = [0,self.length,0,self.width], cmap = 'coolwarm')
        cb = plt.colorbar()
        cb.set_label("Temperature", rotation = 270)
        plt.show()

    def line_plot(self, x = 13):
        #Plots just the central row of the rod
        nodes = np.arange(self.y_nodes)
        temps = self.temp_field[x]

        plt.plot(nodes,temps)
        plt.show()

    def make_matrices(self):
        #The diffusion is going to modeled by seeing the heat diffused in/out of
        # any node in both the x and y direction, so both a y and x equation is
        # needed because h will be different for them both.

        #The coeffecients
        Cx = self.alpha*self.time_step/(self.x_step**2)
        Cy = self.alpha*self.time_step/(self.y_step**2)

        #Setting up 0 matrices to fill the diagnols
        x_matrix = np.zeros((self.x_nodes,self.x_nodes))
        y_matrix = np.zeros((self.y_nodes,self.y_nodes))

        #Filling the central diagnol
        np.fill_diagonal(x_matrix, (1+2*Cx))
        np.fill_diagonal(y_matrix, (1+2*Cy))

        #Setting up the bottom diagnol
        np.fill_diagonal(x_matrix[1:], -Cx)
        np.fill_diagonal(y_matrix[1:], -Cy)

        #Setting up the top diagnol
        np.fill_diagonal(x_matrix[:,1:], -Cx)
        np.fill_diagonal(y_matrix[:,1:], -Cy)

        #Setting the top and bottom rows to the appropiate values
        x_matrix[0][0] = x_matrix[-1][-1] = 1 + Cx
        y_matrix[0][0] = y_matrix[-1][-1] = 1 + Cy

        #Storing only the decompositions
        self.LU_x = spla.lu_factor(x_matrix)
        self.LU_y = spla.lu_factor(y_matrix)

    def iterate(self):
        #Calculating the temperature diffusing in or out of nodes in the x and y
        #direction. Then storing new temperatures.

        #Empty to store the results of diffusion in the x and y direction
        row_diffuse = np.zeros((self.y_nodes,self.x_nodes))
        column_diffuse = np.zeros((self.y_nodes,self.x_nodes))

        #For every row (x-direction) treating it as a single rod
        for i,row in enumerate(self.temp_field):
            row_update = spla.lu_solve(self.LU_x, row)
            row_update[-1] = row_update[-2]
            row_diffuse[i] = row_update

        #For every column (y-direction) treating it as a single rod
        for j, col in enumerate(self.temp_field.transpose()):
            col_update = spla.lu_solve(self.LU_y, col)
            col_update[-1] = col_update[-2]
            col_update[0] = col_update[1]
            column_diffuse[:,j] = col_update

        #Since row_diffuse and column diffuse = x or y diffusion + original
        # to get totl diffusion + original we need to do x + y diffusion - original
        total_diffuse = row_diffuse + column_diffuse - self.temp_field
        total_diffuse[:,0][int(self.y_nodes/4): int(self.y_nodes*(3/4))] = 1000
        self.temp_field = total_diffuse.copy()

        self.time += self.time_step

    def iterate_to(self,max_time):
        #Iterates to the time required
        while self.time < max_time:
            self.iterate()

    def iterate_to_converge(self):
        #Iterates until the change was less then 1 degree
        old = self.temp_field.copy()
        check = 1000
        while check > 1:
            self.iterate()
            check = np.sum(np.abs(old-self.temp_field))
            old = self.temp_field.copy()


"""--------------------------------Functions------------------------------------"""

def quick_test():
    #Function to check boundaries are being set correctly could really be binned
    # but if things break will be thankful I never did

    #Ive left this here for sentimental reasons mostly
    Test = Field()
    Test.set_boundary(10,(5,5),(5,20))
    Test.set_boundary(-10,(20,5),(20,20))
    Test.quick_plot()

"""Capacitor functions"""

def make_capacitor(seperation, width, charge, load_guess = True, boo = False, small = False, demonstration = False):
    #Generates a field with boundary conditions of a capacitor

    Capacitor = Field((1,1))
    if small:
        Capacitor.x_nodes = 50
        Capacitor.y_nodes = 50
    #The potential difference between the plates being saved
    Capacitor.V = np.abs(charge)*2

    #Expanding the plot out in x and y if needed for the desired width and
    # seperation
    if seperation > Capacitor.y_nodes:
        Capacitor.y_nodes *= mt.ceil(seperation/Capacitor.y_nodes)

    if width > Capacitor.y_nodes:
        Capacitor.x_nodes *= mt.ceil(width/Capacitor.x_nodes)

    #Applies previous changes
    Capacitor.reset()

    #Defining the rows the plates are on
    plate_1 = int((Capacitor.y_nodes - seperation)/2)
    plate_2 = plate_1 + seperation

    #Defining the sizes of the plates
    edge_1 = int((Capacitor.x_nodes - width)/2)
    edge_2 = edge_1 + width

    #Load a guess of linear rise of charge 0 to the charge of the first plate
    # up to the first plate then between the 2 plates a linear relationship
    if load_guess:
        for i in range(plate_1):
            Capacitor.scalar_field[i] = i/plate_1 * charge
        for i in range(seperation):
            Capacitor.scalar_field[i + plate_1] = charge - 2*charge*i/seperation
        for i in range(plate_2-seperation+1):
            Capacitor.scalar_field[-i] = i/(plate_2-seperation) * -charge

    #Setting the boundaries after everything else to ensure they have not been
    # overwritten
    Capacitor.set_boundary(charge, (plate_1,edge_1), (plate_1,edge_2))
    Capacitor.set_boundary(-charge, (plate_2,edge_1), (plate_2, edge_2))

    #Dummy arguement for checking the capacactiros were set up correcty
    if boo:
        Capacitor.quick_plot()

    #Prints and plots data for the demonstration in the menu
    if demonstration:
        Capacitor.iterate_to_converge()
        Capacitor.plot_field()

        E_exp,mean = inside_check(Capacitor)
        outside = outside_check(Capacitor)
        print("The average outside value was ", outside)
        print("The difference between the expected and inside value is ", (E_exp-mean))
        print("It took ", Capacitor.count, " iterations to converge")

    #BEHOLD THE ALMIGHTY CREATED CAPACITOR
    return Capacitor

def get_inside(Capacitor):
    #Extracting the region bounded by the plates in a capacitor, should only be
    # be used with a capacitor made with make_capacitor

    Capacitor.E_field_get()

    #The boundaries are fed in top plate, bottom plate so this extraction can be
    # performed
    top_left = Capacitor.boundaries[0]
    bottom_right = Capacitor.boundaries[-1]

    min_x, min_y = top_left
    max_x, max_y = bottom_right

    #Slicing the E field for the region inside
    inside = Capacitor.E_field[min_x+1:max_x][:,min_y+1:max_y]

    return inside

def inside_check(Capacitor, only_inner = False):
    #Compares the E field inside to the expected value from E = V/d

    #Generating the E_field if not already done
    if 'E_field' not in Capacitor.__dict__:
        Capacitor.iterate_to_converge()
        Capacitor.E_field_get()

    inside = get_inside(Capacitor)

    #Can only examine the most central region of the capacitor
    if only_inner:
        start = int(0.25*len(inside))
        end = int(0.75*len(inside))
        inside = inside[:,start:end]
    #Faster way of getting mean and std
    inside_1d = inside.ravel()

    mean = np.mean(inside_1d)

    #Distance taken as the middle of one plate to middle of the other
    E_exp = Capacitor.V/(len(inside)+1)

    return E_exp, mean

def outside_check(Capacitor, check_region = False, print_return = 'return'):
    #Extracts the region outside the capacitor and compares it to the 0 expected

    #Generating the E_field if not already done
    if 'E_field' not in Capacitor.__dict__:
        Capacitor.iterate_to_converge()
        Capacitor.E_field_get()

    #Extraacting the region edges like previously
    top_left = Capacitor.boundaries[0]
    bottom_right = Capacitor.boundaries[-1]

    min_x,min_y = top_left
    max_x,max_y = bottom_right

    #Setting up the sum and count to find the mean later
    sum_outside = 0
    count = 0

    #Check_region will lead to plotting the area excluded from the mean
    if check_region:
        inside_check = np.zeros_like(Capacitor.E_field)

    #Iterates across every value, if the multi index is within the region it is
    #skipped. Else it is added to the sum and counted
    for i, row in enumerate(Capacitor.E_field):
        for j, element in enumerate(row):

            if min_x-1 < i and max_x+1 > i and min_y-1 < j and max_y+1 > j:
                continue

            #for the plot inside check
            if check_region:
                inside_check[i][j] = 10

            sum_outside += element
            count += 1

    if check_region:
        plt.imshow(inside_check)
        plt.show()

    #Averaging all the outside
    outside_average = sum_outside/count
    #Since a capacitor holding a 1000Vm^-1 field with 0.3Vm^-1 leakage is better
    # than one with 10 Vm^-1 but the same leakage
    relitive_outside = outside_average / np.max(Capacitor.E_field)

    #Printing if in my demonstration later or returning if used in analysis later
    if print_return == 'print':
        print("The mean field outside is ", sum_outside/count)
        print("As a percentage of inside field this is ", relitive_outside)
    if print_return == 'return':
        return relitive_outside

def plot_inside(Capacitor):
    #Showing a 3d plot of the E field within the region

    #Setting up the figure and axis
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #Finding the difference between expected and observed field stregnth
    inside = get_inside(Capacitor)
    residual = inside-(Capacitor.V/(len(inside)+1))

    #ax.plot_surface requires this meshgrid
    x = np.arange(len(inside[0]))
    y = np.arange(len(inside))
    x,y = np.meshgrid(x,y)

    #Finding the largest deviance from 0, so can set the color map to be
    # symetric around zero but encompass all points.
    upper_limit = np.abs(np.max(residual))
    lower_limit = np.abs(np.min(residual))

    largest_limit = max([upper_limit,lower_limit])

    surf = ax.plot_surface(x,y, residual, cmap = 'seismic', vmin = - largest_limit,
                           vmax = largest_limit)


    cb = fig.colorbar(surf)
    cb.set_label("Difference from V/d", rotation = 270)
    plt.show()

def inside_plot():
    #Test a bunch of different capacitors at differnt seperation and widths
    seperation = [5,10,15,20,25,30,35,40,45]
    width = [5,10,15,20,25,30,35,40,45]

    #Output the capacitors in array of capacitors and their respective ad Values
    # in a similar array
    Cap_array = []
    ad_array = []

    #Iterating over every combination of seperation and width
    for sep in seperation:
        #Rows have the same seperation
        Cap_list = []
        AD_list = []
        for wid in width:

            #Keeps the expected field constant 10s
            Cap = make_capacitor(sep,wid,10*sep,small = True)
            Cap.iterate_to_converge()

            Cap_list.append(Cap)
            AD_list.append(wid/sep)
        #Adding the Capacitors and ad to the array
        Cap_array.append(Cap_list)
        ad_array.append(AD_list)

    #Making arrays to store the inside values and outside values
    inside_array = np.zeros_like(Cap_array)
    outside_array = np.zeros_like(Cap_array)

    #For every capacitor finding the inside and outside deviance from expected
    for i,row in enumerate(Cap_array):
        for j, field in enumerate(row):
            E_exp, mean = inside_check(field)
            inside_array[i][j] = E_exp - mean
            outside_array[i][j] = outside_check(field)

    for inside_row, ad_row, sep in zip(inside_array, ad_array,seperation):
        plt.plot(ad_row, inside_row, label = ('Separation of ' + str(sep)))

    plt.xlabel("a/d ratio")
    plt.ylabel("V/d - mean inside value")
    plt.legend()
    plt.show()

    for outside_row, ad_row, sep in zip(outside_array, ad_array,seperation):
        plt.plot(ad_row, outside_row, label = ('Separation of ' + str(sep)))

    plt.xlabel("a/d ratio")
    plt.ylabel("Mean outside value")
    plt.legend()
    plt.show()


"""Other field tests"""

def gauss_jacobi_result():
    #Generates 2 idential capacitors and converges them both for both jacobi,
    # gauss then compares there differences. And comparing their E fields.

    #Making 2 of the same capacitor with the different methods
    gauss = make_capacitor(40,40,10)
    gauss.method = 'gauss_a'

    jacobi = make_capacitor(40,40,10)
    jacobi.method = 'jacobi_a'

    #Iterating them both and getting the E fields
    gauss.iterate_to_converge()
    jacobi.iterate_to_converge()

    gauss.E_field_get()
    jacobi.E_field_get()

    #An array of the differences between the models
    difference = gauss.E_field - jacobi.E_field

    max = np.max(difference)
    min = np.min(difference)

    largest= np.max([np.abs(max), np.abs(min)])


    #Relative differences
    difference_rel = np.abs(difference)/np.abs(gauss.E_field)

    #Sum total differences
    rel_difference = np.sum(np.abs(difference))/np.sum(np.abs(gauss.E_field))

    plt.imshow(difference, origin = 'lower',cmap = 'seismic',vmin = -largest, vmax= largest)
    cb = plt.colorbar()
    cb.set_label("Difference")
    plt.show()

    print("The relitive difference is :", round(rel_difference,3)*100, "%")

def gauss_jacobi_speed(low, high, step, returns = False):
    #Tests the time taken and number of iterations for arrays of different sized
    # fields with a single potential along the central row.

    #Lists for the time to plots to be stored
    fields = []
    gauss_time = []
    gauss_a_time = []
    jacobi_time = []
    jacobi_a_time = []

    #Generates the fields in a list of field objects
    for nodes in range(low,high,step):
        Test = Field((1,1))

        Test.x_nodes = nodes
        Test.y_nodes = nodes

        Test.reset()

        #Has to be integer inputs
        x = int(nodes/2)

        Test.set_boundary(10,(x,0), (x,nodes-1))

        fields.append(Test)

    #Generates identical fields for all methods to be tested on
    fields_2 = deepcopy(fields)
    fields_3 = deepcopy(fields)
    fields_4 = deepcopy(fields)

    #Explicitly setting all methods
    for field in fields:
        field.method = 'gauss'

    for field in fields_2:
        field.method = 'jacobi'

    for field in fields_3:
        field.method = 'jacobi_a'

    for field in fields_4:
        field.method = 'gauss_a'

    #Timing each methods speeds to converge and append it to the list
    for field in fields:
        t0 = time.time()
        field.iterate_to_converge()
        t1 = time.time()
        gauss_time.append(t1-t0)
        print("Node number ", field.x_nodes)

    for field in fields_2:
        t0 = time.time()
        field.iterate_to_converge()
        t1 = time.time()
        jacobi_time.append(t1-t0)
        print("Node number ", field.x_nodes)

    for field in fields_3:
        t0 = time.time()
        field.iterate_to_converge()
        t1 = time.time()
        jacobi_a_time.append(t1-t0)
        print("Node number ", field.x_nodes)

    for field in fields_4:
        t0 = time.time()
        field.iterate_to_converge()
        t1 = time.time()
        gauss_a_time.append(t1-t0)
        print("Node number ", field.x_nodes)

    #Generates lists of iteration numbers
    gauss_iterations = [field.count for field in fields]
    jacobi_iterations = [field.count for field in fields_2]
    jacobi_a_iterations = [field.count for field in fields_3]
    gauss_a_iterations = [field.count for field in fields_4]

    nodes = [i for i in range(low,high,step)]

    plt.plot(nodes, gauss_iterations, label = 'Gauss')
    plt.plot(nodes, jacobi_iterations, label = 'Jacobi')
    plt.plot(nodes, jacobi_a_iterations, label = 'Accelerated Jacobi')
    plt.plot(nodes, gauss_a_iterations, label = 'Accelerated Gauss')
    plt.xlabel("Nodes along 1 edge")
    plt.ylabel("Iterations to converge")
    plt.legend(loc = "best")
    plt.show()

    plt.plot(nodes, gauss_time, label = 'Gauss')
    plt.plot(nodes, jacobi_time, label = 'Jacobi')
    plt.plot(nodes, jacobi_a_time, label = 'Accelerated Jacobi')
    plt.plot(nodes, gauss_a_time, label = 'Accelerated Gauss')
    plt.xlabel("Nodes along 1 edge")
    plt.ylabel("Time to converge")
    plt.legend(loc = "best")
    plt.show()

    if returns:
        return gauss_time,jacobi_time,gauss_a_time,jacobi_time,gauss_iterations,jacobi_iterations, gauss_a_iterations,jacobi_a_iterations

def potential_difference(gradient):
    #Testing a field for a potential gradient across the entire region, expected
    # is a constant field and a linear potential change

    #Makes the field, extracting max_x and max_Y
    Potential_difference = Field((1,1))
    max_x = Potential_difference.x_nodes-1
    max_y = Potential_difference.y_nodes-1

    #Sets the boundary conditions as first and last rows
    Potential_difference.set_boundary(gradient/2, (0,0), (0,max_y))
    Potential_difference.set_boundary(-gradient/2, (max_x,0), (max_x,max_y))

    Potential_difference.iterate_to_converge()

    #Finds and plots the field
    Potential_difference.E_field_get()
    Potential_difference.plot_field()

    #Extracts the column and finds the errors on that, ensures nodes will be the
    # right size.
    E_value_col = Potential_difference.E_field[:,19]
    E_errors = []
    nodes = []
    for i,value in enumerate(E_value_col):
        error = np.abs(value*0.06)
        if error < 0.03:
            error = 0.03
        E_errors.append(error)
        nodes.append(i)

    #Generates the exepcted field as a list of equal length
    E_exp = gradient/max_y
    E_exp = [E_exp]*(Potential_difference.y_nodes)

    #Calculates the chisquare value
    chi_squared = 0
    for obs,exp,err in zip(E_value_col,E_exp,E_errors):
        chi_squared += (obs-exp)**2/err

    #Print statements and graphs
    print("The chi squared per degree value is ", chi_squared)
    print("The mean of the values is ", np.mean(E_value_col))
    print("The expected value is ", E_exp[2])

    plt.errorbar(nodes,E_value_col,yerr = E_errors, label = "Model value's")
    plt.plot(nodes, E_exp, label = 'E = V/D prediction')
    plt.xlabel("Nodes")
    plt.ylabel("E Field Strength")
    plt.legend(loc = 'upper center')
    plt.show()

def poisson_check():
    #Generates a random field and double diffentiates it to check this = 0

    #Creating a field
    Poisson_check = Field((1,1))

    #Makes these variable names smaller for readablity sake
    max_x, max_y = Poisson_check.x_nodes, Poisson_check.y_nodes

    #Chooses random row and columns which are not near the edge
    random_row = randint(10, max_y - 10)
    random_col = randint(10, max_x - 10)

    #Generates random values for the row and column to have
    row_value = randint(10,20)
    col_value = randint(10,20)

    #Sets the boundaries based on the randomly generated numbers
    Poisson_check.set_boundary(row_value, (random_row,0), (random_row, max_x-1))
    Poisson_check.set_boundary(-col_value, (0,random_col), (max_y-1, random_col))

    Poisson_check.iterate_to_converge()

    Poisson_check.E_field_get()

    #Differentiates the found dx and dy to get the double differentials
    dydx, dxdx = np.gradient(Poisson_check.dx)
    dydy, dxdy = np.gradient(Poisson_check.dy)

    #Formula for the laplace in cartesian coordinate
    laplace = dxdx+dydy

    #Skips over the rows around the boundary conditions where the laplacian
    # breaks
    split_half = laplace[:random_row-2]
    split_half_2 = laplace[random_row+3:]
    laplace_trimmed = np.concatenate((split_half,split_half_2))

    #Skipping the edges aswell where the laplacian also breaks
    laplace_trimmed = laplace_trimmed[2:-2].transpose()

    #Skipping over the columns
    split_half = laplace_trimmed[:random_col-2]
    split_half_2 = laplace_trimmed[random_col+3:]
    laplace_trimmed = np.concatenate((split_half,split_half_2))

    #Skipping over the edges
    laplace_trimmed = laplace_trimmed[2:-2].transpose()

    #Finding the 2 max values
    max_laplace = np.max(laplace_trimmed)
    max_E = np.max(np.abs(Poisson_check.E_field))

    print("The max value of the laplace (ignoring around boundary coniditions) is ", max_laplace)
    print("The max value of the field is ", max_E)

    #Plotting teh whole laplcaian and then the laplacian with edge nodes removed
    print("The full laplace")
    plt.imshow(laplace, cmap = 'coolwarm')
    plt.xlabel("X Nodes")
    plt.ylabel("Y Nodes")
    cb = plt.colorbar()
    cb.set_label("Potential (V)", rotation=270)
    plt.show()

    print("The laplace minus the area around boundary conditions")
    plt.imshow(laplace_trimmed, cmap = 'coolwarm')
    plt.xlabel("X Nodes")
    plt.ylabel("Y Nodes")
    cb = plt.colorbar()
    cb.set_label("Potential (V)", rotation=270)
    plt.show()

    print("The electric potential")
    Poisson_check.quick_plot()

    #Generates an array for the errors to be inserted into.
    error_array = np.zeros((max_x, max_y))

    #iterates over the grid
    for i in range(max_x):
        for j in range(max_y):
            #Finds the errors
            error = 0.06 * Poisson_check.E_field[i][j]
            #Inserting the error flooor
            if error < 0.03:
                error = 0.03
            #Adding the positive value of the array
            error_array[i][j] = np.abs(error)

    #Relative errors rather than absolute
    value_error = np.abs(laplace) / error_array

    #Removing those dodgy values again
    split_half = value_error[:random_row-2]
    split_half_2 = value_error[random_row+3:]

    value_error = np.concatenate((split_half,split_half_2))[2:-2].transpose()

    split_half = value_error[:random_col-2]
    split_half_2 = value_error[random_col+3:]

    value_error = np.concatenate((split_half,split_half_2))[2:-2].transpose()

    #Finding maximum number of deviations away

    max_deviations_away = np.max(value_error)

    print("The maximum number of standard deviations a point away from zero is ", max_deviations_away)

def method_images_test():
    #Generates 2 fields which should be the same in the rhs of the plot

    Charges_and_plates = Field((1,1))

    #Sets the charge
    Charges_and_plates.set_boundary(10,(12,70),(88,70))
    #Sets the grounded plate
    Charges_and_plates.set_boundary(0, (0,50),(99,50))

    print("Begining first iteration")
    Charges_and_plates.iterate_to_converge()

    Charges_and_charges = Field((1,1))

    #Sets the 2 charged areas
    Charges_and_charges.set_boundary(10,(12,70),(88,70))
    Charges_and_charges.set_boundary(-10,(12,30),(88,30))

    print("Begining second iteration")
    Charges_and_charges.iterate_to_converge()

    #Changing names for lines sakes
    A = deepcopy(Charges_and_charges)
    B = deepcopy(Charges_and_plates)

    A.E_field_get()
    B.E_field_get()

    #Taking only the RHS of the plots
    A_half = A.E_field[:,int(len(A.E_field)/2)+1:]
    B_half = B.E_field[:,int(len(B.E_field)/2)+1:]

    print("The 2 plates with the potential and grounded plate on the left and the 2 potentials on the right")

    #Plotting the 2 halves side by side
    plt.subplot(1,2,1)
    plt.imshow(A_half)

    plt.subplot(1,2,2)
    plt.imshow(B_half)
    cb = plt.colorbar()
    cb.set_label("E field (Vm^-1)", rotation = 270, labelpad = 10)

    plt.show()

    #The difference between them
    abs_err = np.abs(A_half-B_half)
    mean_percentage_diff = np.mean(abs_err) * 100

    A.plot_field()
    B.plot_field()

    print("The mean difference between the 2 models was ", round(mean_percentage_diff,2),"%")

def point_source():
    #Generates a point source
    point_field = Field((1,1))

    #Needs the node array to divide through by
    point_field.make_node_array()

    print("Once again this takes forever apologies")
    #Iterates 5000 time adding the rho(x,y) point to the central part to simulate
    # the point source
    for a in range(5000):
        point_field.grid_iterate_jacobi_alt()
        point_field.scalar_field[50][50] += 100

    #Plots the field
    plt.imshow(point_field.scalar_field)
    plt.show()

    #Takes only the central row of the E field.
    point_field.E_field_get()

    #Takes only the first half of the middle row
    E_field_values = point_field.E_field[50][:50]
    distance = [50-i for i in range(50)]

    plt.plot(distance, E_field_values)
    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)
    plt.xlabel("Distance from charge")
    plt.ylabel("Electric field strength")
    plt.show()

    return point_field

def convergence_graph():
    #Keeps a list of the amount of chagne between each iteration

    Cap = make_capacitor(10,10,10, load_guess = False)
    list_check = []
    iterations = []

    #Setting up convergence checks
    Cap.make_node_array()
    Cap.grid_iterate_jacobi_alt()
    Cap.convergence_check()
    Cap.grid_iterate_jacobi_alt()
    Cap.convergence_check()


    print("This section takes a while as the model goes to higher tolerance than\
    other plots, they will be shown at the end")
    #A more strict convergence criteria then previously seen
    while Cap.check > 1e-7:
        #Appends the relevant values
        list_check.append(Cap.check)
        iterations.append(Cap.count)
        #Iterates and updates the chec
        Cap.grid_iterate_jacobi_alt()
        Cap.convergence_check()

    #Plotting the Full data
    print("The full iterations, and change caused by those iterations")
    plt.plot(iterations, list_check)
    plt.xlabel("Number of Iterations")
    plt.ylabel("The sum change of the row and column")
    plt.show()

    #Plots the reduced data to show it continues
    reduced_iterations = iterations[200:]
    reduced_list_check = list_check[200:]

    print("Reduced iterations, and change caused by those iterations")
    plt.plot(reduced_iterations, reduced_list_check)
    plt.xlabel("Number of Iterations")
    plt.ylabel("The sum change of the row and column ")
    plt.show()

    #Logs this information to show exponential relationship
    print("Logarithm")
    log_list_check = np.log(list_check[200:])
    plt.plot(iterations[200:], log_list_check)
    plt.xlabel("Number of Iterations")
    plt.ylabel("The natural logarithm of the sum change")
    plt.show()

    #Values retrieved from origin
    gradient = -0.00703
    intercept = -1.10052

    #Assuming the tail is exponential (y=Ae^-cx) calculates the area of the tail
    #Also assumes a tail starts at 200 for the report a more detailed search
    #was done by size.

    A = np.exp(intercept)
    c = np.abs(gradient)
    tail_start = 200
    tail_error = A/c * np.exp(-tail_start * c)

    #The tail error is the sum of the rrors for all the values on a row and a
    # column. So this finds teh average error for the row and column
    row_column_sum = sum(np.abs(Cap.converge_col) + np.abs(Cap.converge_row))

    fractional_error = round(tail_error/row_column_sum,4)

    print("An estimate of the error is: ", fractional_error*100, "%")

def alt_speed_test(low, high,step,returns = False):
    #Time just alternate accelerate versions of jacobi and gaussself.

    #Generates the nodes
    nodes = np.arange(low,high,step)

    #Generates a list of field objects so both methods are solving the Same
    #fields
    fields = []
    for node in nodes:

        Test_field = Field((1,1))

        #Sets the size of the array
        Test_field.x_nodes = node
        Test_field.y_nodes = node
        Test_field.reset()

        #Taking the central row
        x = int(node/2)

        #Setting the central row value
        Test_field.set_boundary(10,(x,0), (x,node))

        fields.append(Test_field)

    #Copying it so they both have the same
    fields_copy = deepcopy(fields)

    g_times = []
    j_times = []

    #Timing them both
    for field in fields:
        t0 = time.time()
        field.iterate_to_converge()
        t1 = time.time()
        j_times.append(t1-t0)

    for field in fields_copy:
        field.method = 'gauss_a'
        t0 = time.time()
        field.iterate_to_converge()
        t1 = time.time()
        g_times.append(t1-t0)

    #Plots the data
    plt.plot(nodes,g_times,label = 'Gauss')
    plt.plot(nodes,j_times,label = 'Jacobi')
    plt.xlabel("Size of array")
    plt.ylabel("Time to converge")
    plt.legend()
    plt.show()

    if returns:
        return nodes,g_times,j_times,[field.count for field in fields],[field.count for field in fields_copys]


"""Rod functions"""

def compare_rods():
    #Seeing the difference between the 1d rod and 2d rod

    #Making the 2 rods as identical beyond one having thickness and iterating
    # them for the same time
    rod_2 = Rod_2()
    while rod_2.time < 5000:
        rod_2.iterate()

    rod = Rod(1000,50,100)

    while rod.time < 5000:
        rod.take_time_step()

    nodes = [i+1 for i in range(50)]

    step = rod.temp_array[::2][3] - rod_2.temp_field[12][3]
    new_temp = rod_2.temp_field[12] + step

    plt.plot(nodes, rod.temp_array[::2], label = '1D')
    plt.plot(nodes, rod_2.temp_field[12], label = '2D')
    plt.plot(nodes[1:], new_temp[1:],'--', label = '2d scaled')
    #plt.ylim((0,1000))
    plt.xlabel("Node on rod")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

def show_instability():
    #Repeatdly multiplies an initial array by the matrix

    temp_array = np.array([20]*100)
    temp_array[0] = 1000

    #Rather than trying to make the matrix again just stealing it from here
    rod = Rod(1000,20,100)
    matrix = rod.make_iteration_matrix(returns = True)
    nodes = [i for i in range(100)]

    for a in range(251):
        if a % 25 == 0:
            print("Time = ", (a*20))
            plt.xlabel("Node Position")
            plt.ylabel("Node temperature")
            plt.plot(nodes,temp_array)
            plt.show()

        temp_array = np.dot(matrix,temp_array)
        temp_array[0] = 1000

def test_matrix_accuracy():
    #I found this at 16:47 on submission day so dont have time to
    """
    Revisit and change variable names to something less gross
    """
    rod = Rod(1000,20,100)

    matrix = rod.make_iteration_matrix(returns = True)

    old_rod = rod.temp_array.copy()

    residuals_list = []
    times_list = []
    total_temp_differences = []

    while rod.time < 10000:
        rod.take_time_step()

        new_rod = rod.temp_array.copy()
        old_rod_implicit = np.dot(matrix, new_rod)

        differnce = np.sum(np.abs(old_rod_implicit - old_rod))
        residual = differnce/(np.sum(np.abs(new_rod)))
        residuals_list.append(residual)

        temp_differences = new_rod[1:] - new_rod[:-1]

        squared_temp_differences = np.sum((np.sqrt(np.abs(temp_differences))))**2

        total_temp_differences.append(squared_temp_differences)

        times_list.append(rod.time)

        old_rod = new_rod.copy()

    scale_factor = np.max(residuals_list) / np.max(total_temp_differences)

    scaled_squared_temp_differences = [temp*scale_factor for temp in total_temp_differences]

    mean_residual = np.mean(np.abs(residuals_list))

    plt.plot(times_list, scaled_squared_temp_differences, label = 'Size of tempertaure steps')
    plt.plot(times_list, residuals_list, label = 'Residuals')
    plt.xlabel("Time (s)")
    plt.ylabel("Size of residual and scaled error")
    plt.legend()
    plt.show()

    error = simps(residuals_list, dx = 20)

    print("The temperature steps have been scaled by: ", scale_factor)
    print("Top end error estimate is ", round(error,2), "%")
    print("The average residual was ", mean_residual)

def condition_number():
    #Tests the condition number * number of steps needed for 1000 seconds

    time_steps = [0.001,0.01,0.1] + [i for i in range(100)]
    cond_number = []
    cond_number_alt = []

    #Generates the rod and extracts the matrix and calculate the condition number
    for time in time_steps:
        rod = Rod(100, time, 100)
        #finding the condition number of the matrix and adding it to the list
        cond = np.linalg.cond(rod.make_iteration_matrix(returns = True))
        cond_number.append(cond)
        cond_number_alt.append(cond * 1000/time)

    plt.plot(time_steps, cond_number)
    plt.show()

    plt.plot(time_steps[3:], cond_number_alt[3:])
    plt.xlabel("Time steps")
    plt.ylabel("Condition number x number of iterations for 1000s")
    plt.show()

def step_compare(skip_first = True):

    time_steps = [1,2,5,10,20,50,100]
    nodes = [i for i in range(100)]

    for time in time_steps:

        if skip_first:
            skip_first = False
            continue

        rod = Rod(1000,time,100,ice = True)

        while rod.time < 1000:
            rod.take_time_step()

        plt.plot(nodes,rod.temp_array, label = ("Time step = " + str(time) + 's'))
    plt.xlabel("Position in rod")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()


"""Menu code"""

menu = 0
while menu != 'q':
    menu = input("Note: 'Reference row and column created' is printed every time a new iteration to convergence is started.\
     \n\nPlease input your choice of part (either 1,2 or 3 or q to quit):\
     \n 1: Examples of various fields, known cases and demonstrating the poisson equation is satisfied.\
     \n 2: Showing the potential and field by varying sized capacitors, then graphs showing the behaviour of the field for varying a/d\
     \n 3: Showing the way a rod disperses temperature\
     \n Equally input fig1/fig2.. ect to generate that plot from the report\
     \n Please enter your selection: ")

    if menu == '1':
        print("\n\nA demonstration of the point source field including the +pho term, taken as an arbitarirly high number")
        point_source()
        input("Please hit enter to continue: ")

        print("\n\nA field with just a potential difference across the entire region")
        potential_difference(99)
        input("Please hit enter to continue: ")

        print("\n\nShowing the method of images result of the boundaries conditions defining a field")
        method_images_test()
        input("Please hit enter to continue: ")

        print("\n\nA capacitor example, more are shown in option 2")
        Cap = make_capacitor(30,30,60, demonstration = True)
        input("Please hit enter to continue: ")

        print("\n\nCheck for the Poisson equation is satsified")
        poisson_check()
        input("Please hit enter to continue: ")

    if menu == '2':
        print("\n\nCapacitor 1-Very good one, 5 seperation 90 width")
        Cap = make_capacitor(5,90,100,demonstration=True)
        input("Please hit enter to continue: ")

        print("\n\nCapcitor 2-Not as good, 70 width 15 seperation")
        Cap = make_capacitor(15,70,100,demonstration=True)
        input("Please hit enter to continue: ")

        print("\n\nCapacitor 3-An okay one, 50 width 50 seperation")
        Cap = make_capacitor(50,50,100,demonstration=True)
        print("\nThe inside looks like this")
        plot_inside(Cap)
        input("Please hit enter to continue: ")

        print("\n\nCapcitor 4-A bad one, 30 width 70 seperation")
        Cap = make_capacitor(70,30,100,demonstration=True)
        print("\nThe inside looks like this")
        plot_inside(Cap)
        input("Please hit enter to continue: ")

        print("\n\nCapcitor 5-A pretty awful one, 10 width 90 seperation")
        Cap = make_capacitor(90,10,100,demonstration=True)
        input("Please hit enter to continue: ")

        print("The a/d graphs look like these")
        inside_plot()
        input("Please hit enter to return to the menu: ")

    if menu == '3':
        print("\n\nDemonstrates the instability with the forward euler method")
        show_instability()
        input("Please hit enter to continue: ")

        print("\n\nRod in ice, several different time steps plotted")
        rod = Rod(1000,50,100)
        rod.step_plot(10000,1000)
        input("Please hit enter to continue: ")

        print("\n\nRod in ice, tracking the temperature of nodes")
        rod = Rod(1000,50,100, ice = True)
        rod.full_plot(10000)
        input("Please hit enter to continue: ")

        print("\n\nRod not in ice, tracking tmperature of nodes")
        rod = Rod(1000,50,100, ice = False)
        rod.full_plot(10000)
        input("Please hit enter to continue: ")

        print("\n\nComparing step size with and then without the 1s step")
        step_compare(skip_first=True)
        step_compare(skip_first=False)
        input("Please hit enter to continue")

        print("\n\nThe 2d rod can be seen here")
        rod_2 = Rod_2()
        rod_2.quick_plot()
        input("Please hit enter to continue: ")

        print("\n\nThe rod after a little bit of iteration")
        rod_2.iterate_to(1000)
        rod_2.quick_plot()
        input("Please hit enter to continue")

        print("\n\nThe rod afte a bit more iteration")
        rod_2.iterate_to(10000)
        rod_2.quick_plot()
        input("Please hit enter to continue")

        print("\n\nA comparisson of the 2d and 1d rod")
        compare_rods()
        input("Please hit enter to return: ")

    if menu == 'fig1':
        convergence_graph()

    if menu == 'fig2':
        print("Here is a smaller version non-log plotted the data from the graph was collated over 2 nights and saved")
        gauss_jacobi_speed(5,30,5)

    if menu == 'fig3':
        gauss_jacobi_result()

    if menu == 'fig4':
        poisson_check()

    if menu == 'fig5':
        potential_difference(99)

    if menu == 'fig6':
        method_images_test()

    if menu == 'fig7':
        Cap = make_capacitor(30,100,60)
        Cap.iterate_to_converge()
        mean,expected = inside_check(Cap)
        print("The difference between expected and mean is ", round((mean-expected),5))
        print("The number of iterations to converge was ", Cap.count)

    if menu == 'fig8' or menu == 'fig9':
        inside_plot()

    if menu == 'fig10':
        Cap = make_capacitor(30,30,60)
        Cap.iterate_to_converge()
        plot_inside(Cap)
        print("The number of iterations to converge was ", Cap.count)

    if menu == 'fig11':
        rod = Rod(1000,20,100, ice = True)
        rod.step_plot(3000,500)

    if menu == 'fig12':
        rod = Rod(1000,20,100)
        outputs = [rod.alpha_explore()]

    if menu == 'fig13':
        condition_number()

    if menu == 'fig14':
        rod = Rod(1000,50,100, ice = False)
        rod.step_plot(25000,1500)

    if menu == 'fig15':
        rod = Rod(1000,20,100, ice = False)
        rod.iterate_to_converge()
        rod.quick_plot()

    if menu == 'fig16':
        rod_2 = Rod_2()
        rod_2.iterate_to(25000)
        rod_2.quick_plot()

    if menu == 'fig17':
        compare_rods()
