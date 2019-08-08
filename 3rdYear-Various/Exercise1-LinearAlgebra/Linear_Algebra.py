"""
Make sure to cast arrays as floats in arrays your putting as the elements in the
matrix class.
"""


"""
A quick guide on using my code:

Setting a matrix or vector:
2 main options, 1 random_float_matrix/random_integer_matrix (same works with float),
or your own.
A = random_float_matrix(3) Produces a 3x3 matrix object full of random floats
A = Matrix(np.array([[youre] ,[elements], [in here]]))
c = random_float_vector(3) produces a 1x3 vector object full of random floats
c = Vector([your, elements in, here])

With the matrix you can run A.get_determinant() to find its determinant which
is stored at A.determinant, A.Get_determinant(matrix) gets the determinant of
the input matrix.

You can set a matrix to be multiplied by a scalar by setting A.scalar or by
casting a scalar arguement when the matrix is initialised ie.
A=Matrix(elements = some array, scalar = 3) to have a 3 outisde. This can be
put inside the matrix by A.scalar_in() or some scalar (d) can be factorised outisde
by A.scalar_out(d). This was mainly used during checks.

The matrix can be inverted by A.invert with the inverse stored at A.inverse as its
own matrix object. A.inverse.scalar_in() is often needed to be cast.

The lu and svd decomposition can be acessed through the functions defined, and
important A.svd_invert() needs to be performed to use sigma cross and U transpose
and V.

For vector with the same V.scalar_in() and V.scalar_out() functions Aswell as
the option to apply a matrix to the vector.
c = V.apply_matrix(A) sets c as a vector obejct equal to Ac.

solve_simul(A,c), lu_solve(A,c), svd_solve finds x for Ax=c

Equivalently 2 matrices can be multipled by matrix_multiplication(A,B) returning
C a matrix object equal to AB.

The pitch carries the geometry of the pitch and the position of the camera and
cables. Normally loading the pitch with its deafult and using that is sufficient
ie:

pitch = Pitch()
plot_tension_map(pitch, other arguements)

in order to change the position of the cable ends pitch.cable_ends has to be
manually overwritten by a list of tuples of the coordinates of the cable ends.
[(bottom left),(bottom right),(top)] they have to be inputted in that order.

pitch.make_matrix_form() makes the matrix and vector for the matrix form of
the simultaneous equations for the system in equillibrium. And stores them in
pitch.tension_matrix and pitch.force_vector.

pitch.axis_centre((x,y)) changes the coordinates so that x,y is now at the origin.
mostly called within other functions.

pitch.angle((x1,y1), (x2,y2)) finds the angle between x1,x2 and y1,y2. mostly
used in pitch.camera_check

pitch.camera_check checks the cameras position to see if its within the triangle
of cable ends to make sure it can indeed be supported by the cables.


"""
import numpy as np
import random as rd
import time
import scipy.linalg as spla
import os
import matplotlib.pyplot as plt
import pandas as pd

"""
Classes
"""

class Matrix():
    """A matrix object, contains the matrix and information about the
    matrix. Aswell as functions such as invert along with storing the result.
    The data is stored in a way where you can have a simpler matrix stored as
    the elements and a scalar also present"""

    def __init__(self, elements = False, scalar = False):
        #Defining a matrix, the elements should be passed in as a numpy
        #ndarray, scalar will be set as 1 if none is given. boo is an arguement
        #used for bug fixing, and testing.

        self.elements = elements
        self.determinant = False
        self.inverse = False
        self.lu = False
        self.piv = False
        self.U = False
        self.Ut = False

        #setting scalar to 1 if none is specified
        if not scalar:
            self.scalar = 1
        else:
            self.scalar = scalar

    def scalar_in(self):
        #Multiplies the scalar into the matrix for when needed

        for i,row in enumerate(self.elements):
            for j,element in enumerate(self.elements[i]):
                self.elements[i][j] = element * self.scalar
        self.scalar = 1

    def scalar_out(self,scalar):
        #Take a scalar out of the matrix useful if that simplifies the matrix

        for i,j in enumerate(self.elements):
            for I,J in enumerate(self.elements):
                self.elements[i][I] = float(self.elements[i][I]) / scalar
        self.scalar = self.scalar * scalar

    def Get_determinant(self, matrix):
        #Uses recursion to find the determinant, it reduces the matrix down to
        # needing 2x2 determinants. Then uses the 2x2 determinant definition
        # to caclulate.
        if matrix.shape == (2,2):
            return ((matrix[0][0] * matrix[1][1])-(matrix[1][0] * matrix[0][1]))

        else:
            determinant = 0
            for i in range(len(matrix[0])):
                row_deleted = np.delete(matrix, 0,0)
                minor = np.delete(row_deleted, i, 1)
                #Calls itself to find the determinant of the new smaller matrix
                determinant += (-1) ** i * matrix[0][i] * self.Get_determinant(minor)
            return determinant

    def get_determinant(self):
        #Calls the determinant finder on its own elements to calculate the matrix
        # determinant
        self.determinant = self.Get_determinant(self.elements)

    def invert(self):
        #Inverts the matrix, returns the result as its own matrix with scalar
        # as the 1/determinant and the matrix as the matrix of cofactors
        # transposed

        if not self.determinant:
            self.get_determinant()

        if self.determinant == 0:
            print("No inverse exists")
            return


        #Relatively arbitary lower bound on dodgy results baased on vague
        # observations with a large safety net
        if np.abs(self.determinant) < 0.00001:
            print("Warning small determinant, results may be inaccurate")

        #Uses the usual [[A,B][C,D]] inverted is 1/det [[D,-C],[-B,A]]
        # definition, if the size is 2x2
        if self.elements.shape == (2,2):
            invert = np.array([
                               [self.elements[1][1],
                                -self.elements[0][1]],
                               [-self.elements[1][0],
                                self.elements[0][0]]
                              ])
            self.inverse = Matrix(
                                  elements = invert,
                                  scalar = 1/(self.determinant*(self.scalar**2))
                                 )
        #For an NxN where N > 2, loops through the whole matrix removing the
        # rows and columns then finding the determinant of the minor
        else:
            inverse = np.zeros(shape = self.elements.shape)
            for i,row in enumerate(self.elements):
                for j in range(len(row)):
                    row_deleted = np.delete(self.elements,i,0)
                    minor = np.delete(row_deleted,j,1)
                    #Assigns in pre-transposed locations
                    inverse[j][i] = (-1)**(i+j) * self.Get_determinant(minor)
            scalar = 1/(self.determinant*(self.scalar**self.elements.shape[0]))

            #Assigns a matrix object to the .inverse variable
            self.inverse = Matrix(elements = inverse, scalar = scalar)

    def lu_decompose(self):
        #Using the scipy.linalg package to decompose
        self.lu, self.piv = spla.lu_factor(self.elements)

    def svd_decompose(self):
        #Using the scipy.linalg package to decompose and creat the sigma,U,V
        # matrices
        self.U, self.sigma, self.Vt = spla.svd(self.elements)
        self.sigma = spla.diagsvd(self.sigma, len(self.elements[0]),
                                  len(self.elements[0]))

    def svd_invert(self):
        #Finds sigma cross by inverting the non-zero diagonal elements and
        # taking the transposes of U,V and the sigmacross
        self.Ut = np.transpose(self.U)
        self.V = np.transpose(self.Vt)
        self.sigmacross = self.sigma

        #Loops through the diagonals
        for i in range(len(self.sigma)):
            check = self.sigmacross[i][i]
            if check == 0:
                continue
            self.sigmacross[i][i] = 1/check

        self.sigmacross = np.transpose(self.sigmacross)

class Vector():
    """
    A Vector class, a mildly more convininient way of storing the results and
    input of x and c in the classic matrix system Ax = c.
    """
    def __init__(self, elements, scalar = False):
        #Assigns elements to the ones given and a scalar outside the vector.
        self.elements = elements
        if not scalar:
            self.scalar = 1
        else:
            self.scalar = scalar

    def scalar_in(self):
        #Multiplies in the scalar stored outside the vector, mostly used in
        # checks performed by itself
        for i in range(len(self.elements)):
            self.elements[i] = self.elements[i] * self.scalar
        self.scalar = 1

    def scalar_out(self, scalar):
        #Takes scalars outside of the matrix, once again mostly used by myself
        # for checks
        for i in range(len(self.elements)):
            self.elements[i] = self.elements[i] / scalar
        self.scalar = self.scalar * scalar

    def apply_matrix(self,A):
        #Left multiplies the vector by the matrix given. Used easily calculating
        # A^-1c or other matrix vector multiplacations.

        #Allows for the matrix entered to either be a matrix object or ndarray
        if type(A) is Matrix:
            A.scalar_in()
            A = A.elements

        #Performs the check that the matrix and vectors are of complementary
        # shapes.
        if len(self.elements) != A.shape[0]:
            print("Not possible to apply this matrix, sizes do not match")
            return

        #Uses the rule of matrix multiplication the sum of row * column
        result = np.zeros(len(self.elements))
        for i,row in enumerate(A):
            new_element = 0
            for j,element in enumerate(row):
                new_element += element * self.elements[j]
            result[i] = new_element
        return Vector(result, scalar = (self.scalar))

class Pitch():

    def __init__(self):
        #Dimension of the pitch
        self.length = 100
        self.width = 70
        #Where the cable ends and camera are placed.
        self.cable_ends = [(5,-5),(95,-5),(50,75)]
        self.camera_position = (50,35)
        #Information about the camera
        self.camera_weight = 50
        self.camera_height = 7
        #Changes to false by camera_check if the camera is outside the triangle
        # formed by the cable_ends
        self.camera_supported = True
        #To transfer back to pitch-centric coordinates
        self.corner_position = (0,0)

    def make_matrix_form(self):
        #Creates the matrix that needs to be inverted and sets that as a matrix
        # object at self.tension_matrix, and creates the force vector and places
        # that at self.force_vector

        x_elements = []
        y_elements = []
        z_elements = []

        #Finds the x and y coordinates of the cable ends in the system where then
        # origin is at the camera position. Then appends cos(theta), sin(theta)
        # in a/h and o/h form to the x and y lists.
        for x in self.cable_ends:
            rel_x = self.camera_position[0] - x[0]
            rel_y = self.camera_position[1] - x[1]

            x_y_mod = np.sqrt(rel_x**2 + rel_y**2)
            x_y_z_mod = np.sqrt(x_y_mod**2 + self.camera_height**2)

            x_elements.append((rel_x/x_y_mod))
            y_elements.append((rel_y/x_y_mod))
            z_elements.append((self.camera_height/x_y_z_mod))

        #Creates the objects and places them within the class
        total_matrix = np.array([x_elements,y_elements,z_elements], dtype = 'float_')
        self.tension_matrix = Matrix(elements = total_matrix)
        self.force_vector = Vector(elements = [0,0,-self.camera_weight * 9.81])

    def axis_centre(self, centre):
        #Changes the position of the origin and so changes the x and y values
        # to match this new coordinate system.

        for i,x_y in enumerate(self.cable_ends):
            self.cable_ends[i] = tuple(np.subtract(x_y, centre))

        self.camera_position = np.subtract(self.camera_position, centre)
        self.corner_position = np.subtract(self.corner_position, centre)

    def line_intercept(self, point1, point2):
        #Previous check that I want to try and rewrite but may forget to and
        #not delete this comment but its 1 o clock and im really bored
        # and am finding this amusing.
        x1,y1 = point1[0],point1[1]
        x2,y2 = point2[0],point2[1]
        return y1 - (y2-y1)/(x2-x1)*x1

    def angle(self, point1, point2):
        #finds the line defined by 2 points and the x axis
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        return (np.abs(np.arctan((y2-y1)/(x2-x1))))

    def camera_check(self):
        #Checks to see if the camera is outside the triangled defined by the camera
        #ends. If it fails any test it returns

        #Sets the axis to be pitch centric
        self.axis_centre(self.corner_position)

        #Gathers the x and y values from the cable ends
        y_list = []
        x_list = []
        for x_y in self.cable_ends:
            y_list.append(x_y[1])
            x_list.append(x_y[0])
        #Seperates the camera_position tuple
        camera_x = self.camera_position[0]
        camera_y = self.camera_position[1]

        #Sets the camera support arguement to false if x is outside the width
        # of the triangle
        if min(x_list) < camera_x and max(x_list) > camera_x:
            self.camera_supported = True
        else:
            self.camera_supported = False
            return

        #Sets the camera support arguement to false if y is outside the height
        # of the trianlge
        if min(y_list) < camera_y and max(y_list) > camera_y:
            self.camera_supported = True
        else:
            self.camera_supported = False
            return

        #Finds the angle between the camera and the bottom right and the
        # top and the bottom right
        max_angle_rope_1 = self.angle(self.cable_ends[0], self.cable_ends[2])
        angle_rope_1 = self.angle(self.cable_ends[0], self.camera_position)

        #If the angle to the camera is greater than the angle to the tip
        # sets the support arguement to false and ends the function
        if angle_rope_1 < max_angle_rope_1:
            self.camera_supported = True
        else:
            self.camera_supported = False
            return


        #Performs the same check for the other corner
        max_angle_rope_2 = self.angle(self.cable_ends[1], self.cable_ends[2])
        angle_rope_2 = self.angle(self.cable_ends[1], self.camera_position)

        if angle_rope_2 < max_angle_rope_2:
            self.camera_supported = True
        else:
            self.camera_supported = False
            return


"""
Functions
"""


def random_integer_matrix(shape):
    #Generates a random 3x3 full of integers. Creates a matrix of 0s and then
    # loops around it replacing it with a random one generated by the random
    # module.
    matrix = np.zeros(shape = shape)
    for i,row in enumerate(matrix):
        for j in range(len(row)):
            matrix[i][j] = rd.randint(0,9)
    #Returns it as a matrix object
    return Matrix(elements = matrix, scalar = 1)

def random_float_matrix(shape):
    #Same thing for the integer matrix but with random floats
    matrix = np.zeros(shape = shape)
    for i, row in enumerate(matrix):
        for j in range(len(row)):
            matrix[i][j] = rd.uniform(0,9)
    return Matrix(elements = matrix, scalar = 1)

def random_interger_vector(N):
    #Does the same protocol for inteer matrix for just a vector
    vector = np.zeros(N)
    for i in range(N):
        vector[i] = rd.randint(0,9)
    return Vector(elements = vector)

def random_float_vector(N):
    #Same again for a float vector rather than the integer vector
    vector = np.zeros(N)
    for i in range(len(vector)):
        vector[i] = rd.uniform(0,9)
    return Vector(elements = vector)

def N_random_matrix(shape,N):
    #Produces a list of matrix objects of N length and of set shape
    matrices = []
    for i in range(N):
        matrices.append(random_float_matrix(shape))
    return matrices

def N_random_vector(size,N):
    #Produces a random list of N vectors of a specific size
    vectors = []
    for i in range(N):
        vectors.append(random_float_vector(size))
    return vectors

def matrix_multiplication(A,B):
    #A function defined for testing the produced inverse in the matrix object
    # uses the rules of matrix multiplication and returns the matrix product
    # as a matrix object. It has a test so 2 ndarray or 2 Matrices.

    #Sets A,B to be the ndarray within the matrix object.
    if type(A) is Matrix:
        A.scalar_in()
        A = A.elements
    if type(B) is Matrix:
        B.scalar_in()
        B = B.elements

    #Finds the shape of the 2 ndarrays
    A_shape = A.shape
    B_shape = B.shape

    #Tests the shapes are compatable
    if A_shape[1] != B_shape[0]:
        print("Multiplaction not possible")
        return

    #Sets an empty ndarray of the size of the reuslting matrix
    result = np.zeros(shape=(A_shape[0],B_shape[1]))

    #Adds the sum of the rows * columns and places it in the relevant
    # position
    for i,row in enumerate(A):
        for j in range(len(B[0])):
            column = B[:,j]
            matrix_element = sum([a*b for a,b in zip(row,column)])
            result[i][j] = matrix_element

    #Creates the matrix object
    C = Matrix(elements = result, scalar = 1)
    return C

def time_test(size, N):
    #Tests the time to invert N matrices
    matrices = N_random_matrix(size, N)
    t0 = time.time()
    for i in range(N):
        A = matrices[i]
        A.invert()
    t1 = time.time()
    return t1 - t0

def time_trial(size, N):
    #Times the 3 different methods to invert the same N matrices of  a set
    # size.

    matrices = N_random_matrix((size,size),N)
    vectors = N_random_vector(size, N)

    t0 = time.time()
    for matrix,vector in zip(matrices, vectors):
        solve = solve_simul(matrix,vector)
    t1 = time.time()
    t_me = t1 - t0

    t0 = time.time()
    for matrix,vector in zip(matrices, vectors):
        solve = lu_solve(matrix,vector)
    t1 = time.time()
    t_lu = t1-t0

    t0 = time.time()
    for matrix,vector in zip(matrices, vectors):
        solve = svd_solve(matrix,vector)
    t1 = time.time()
    t_svd = t1-t0

    return t_me, t_lu, t_svd

def get_t_N(n,N):
    #Takes a maximum marix size n, and then generates the time tests
    N_list = []
    t_list = []
    for size in range(2,n):
        print("Repeating")
        shape = (size,size)
        N_list.append(size)
        t_list.append(time_test(shape, N))
    return t_list, N_list

def plot_tn(N_list,t_list):
    #Just plots the previously generated matrix.
    plt.plot(N_list,t_list)
    plt.xlabel("Size of NxN")
    plt.ylabel("Time for 100 x1000/(N+1)!")
    plt.show()

def solve_simul(A,c):
    #Uses the A^-1c = x solution method and the previously defined Functions
    # to find the solutions to a simultaneous equation Ax = c
    if A.inverse == False:
        A.invert()
    #Generates the result as a matrix
    solution = c.apply_matrix(A.inverse)
    solution.scalar_in()

    return solution

def lu_solve(A,c):
    #Solves the system of simultaneous equations using the LU decomposition
    # and solve from the scipy.linalg and generates a vector object so the method
    # of solving produces a consistent type of result.
    A.lu_decompose()
    x = spla.lu_solve((A.lu, A.piv), c.elements)
    X = Vector(elements = x, scalar = 1)
    return X

def svd_solve(A,c):
    #Uses previously defined functions to find the U transpose, V and sigmacross
    # matrices and then multiplies them together to produce xbar

    A.svd_decompose()
    A.svd_invert()

    #Applies the matrices to the vector c then the next next matrix to that
    # result. slightly decreases the amount of calculations needed
    Utc = c.apply_matrix(A.Ut)
    SCUtc = Utc.apply_matrix(A.sigmacross)
    x_bar = SCUtc.apply_matrix(A.V)
    X_bar = Vector(elements = x_bar, scalar = 1)
    return x_bar

def make_dict(N_list,size_list):
    #Makes a dictionary of time taken and number of repeats and matrix size
    # so a pandas dataframe can be easily generated. Runs the time trials
    # itself so just input size and number of repeats wanted.
    t_me_list = []
    t_lu_list = []
    t_svd_list = []

    for N,size in zip(N_list,size_list):
        t_me, t_lu, t_svd = time_trial(size, N)
        t_me_list.append(t_me)
        t_lu_list.append(t_lu)
        t_svd_list.append(t_svd)

        print(N,size)

    data = { 'Cofactors method time' : t_me_list,
             'LU method time' : t_lu_list,
             'SVD method time' : t_svd_list,
             'Number of repeats' : N_list,
             'Size of matrices' : size_list}

    return data

def check_solve(A,x,c):
    #Multiplies Ax and then compares to c
    c_solve = x.apply_matrix(A)
    error = 0
    for solution, answer in zip(c_solve.elements, c.elements):
        #Had previously used the square of these but went deep into the floating
        # point error
        error += np.abs((float(solution) - float(answer)))
    return error

def tension_find(Pitch, lu_svd_cofactor = 'cofactor', print = True):
    #Generates the tension vector by solving the matrix equaition

    #Checks camera can be supported
    Pitch.camera_check()
    if not Pitch.camera_supported:
        if print:
            print("Camera cannot be supported by ropes as outside the area they \
               enclose")
        return
    #Generates the matrix and c vector
    Pitch.make_matrix_form()

    #Solves for tension using the method specified
    if lu_svd_cofactor == 'cofactor':
        tension = solve_simul(Pitch.tension_matrix, Pitch.force_vector)

    elif lu_svd_cofactor == 'lu':
        tension = lu_solve(Pitch.tension_matrix, Pitch.force_vector)

    elif lu_svd_cofactor == 'svd':
        tension = svd_solve(Pitch.tension_matrix, Pitch.force_vector)

    else:
        print("Invalid method chosen")

    return tension

def plot_tension_map(Pitch, rope_index, x_elements, y_elements,
                     lu_svd_cofactor = 'lu', cmap = 'Vega20b_r', point = (-50,-50)):
    #Finds the tension vectors across the pitch and plots the desired elements
    # and so the relevant tensions. Can plot a singular point on the graph.

    #Recentres the pitch so its pitchcentric
    Pitch.axis_centre(Pitch.corner_position)

    #Getting the difference between elements
    x_step = Pitch.length/x_elements
    y_step = Pitch.width/y_elements

    #Generate the array for x and y elements
    x_points = np.arange(0, Pitch.length + x_step, x_step)
    y_points = np.arange(0, Pitch.width + y_step, y_step)

    #Generates an array of the right size to store the tension
    tension_array = np.zeros(shape = (y_elements+1,x_elements+1))

    #Loops through x and y points, cehcking if the camera is supported, leaving
    # the tension as 0 if its not. And reads of just the revelant tension
    # converting the negative force into the positive tension.
    for i,x in enumerate(x_points):
        for j,y in enumerate(y_points):

            Pitch.camera_position = (x,y)
            Pitch.camera_check()

            if not Pitch.camera_supported:
                continue
            Tension = tension_find(Pitch, lu_svd_cofactor = lu_svd_cofactor)

            if rope_index == 3:
                tension_array[j][i] = -sum(Tension.elements)
                continue

            #This fixes the plot to be rotated the expectedd way
            tension_array[j][i] = (-Tension.elements[rope_index])

    #Just plotting informatiom, origin = 'lower' because the 0 point is in the
    # top left for some reason.
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')
    plt.imshow(tension_array, cmap = cmap, origin = 'lower',
               extent = [0, Pitch.length, 0, Pitch.width])
    #Include a point if specified
    if point != (-50,-50):
        plt.plot(point[0],point[1],'wx', markersize = 12, mew = 2.5, linewidth = 2)
    plt.colorbar(shrink = 0.8, aspect = 5)
    plt.show()

def plot_near_singular(k_values = [], multi_no = 'multi'):
    #Takes either new k_values or the deafult and can solve different simultaneous
    # with those k coeffeceints
    if len(k_values) == 0:
        k_values = [1,0.5,0.2,0.1,0.05,0.01,0.001,1e-4,1e-5,1e-6,1e-7,
                    1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15]
        if multi_no == 'no':
            k_values += [1e-16, 1e-17, 1e-18]

    #Both sets of equations have the same c
    c = Vector(np.array([3,10,13]))

    #Tests each method for the error between the solution and the true relation
    cofactor_err = []
    lu_err = []
    svd_err = []
    for k in k_values:
        #Tests each method for the relevant matrix and solution combination
        #and appends the infromation to the list
        if multi_no == 'multi':
            A = Matrix(elements = np.array([[1,1,1],[1,2,-1],[2,3,k]], dtype = "float_"))
        if multi_no == 'no':
            A = Matrix(elements = np.array([[1,2,5],[3,1,0],[3,1,k]], dtype = "float_"))
        x = solve_simul(A,c)
        cofactor_err.append(check_solve(A,x,c))
        x = lu_solve(A,c)
        lu_err.append(check_solve(A,x,c))
        x = svd_solve(A,c)
        svd_err.append(check_solve(A,x,c))

    #The rest is just staggered plotting information, the input just Allows
    #the user to not have all the plots suddenly thrown at them.
    plt.plot(k_values, cofactor_err, color = 'blue', label = "Cofactor error")
    plt.plot(k_values, lu_err, color = 'red', label = "LU error")
    plt.plot(k_values, svd_err, color = 'green', label = "SVD error")
    plt.xlabel('Values of k')
    plt.ylabel('Residual squared on solution')
    plt.suptitle('Errors on solution as k approaches 0')

    plt.legend()
    plt.xscale('log')

    plt.show()

    input("Hit enter to remove the cofactor line: ")

    plt.plot(k_values, lu_err, color = 'red', label = "LU error")
    plt.plot(k_values, svd_err, color = 'green', label = "SVD error")
    plt.xlabel('Values of k')
    plt.ylabel('Residual squared on solution')
    plt.suptitle('Errors on solution as k approaches 0')

    plt.legend()
    plt.xscale('log')

    plt.show()

    input("Hit enter too see a log-log plot: ")

    #Creates new lists which have the 0 points removed due to them being plotted
    # at minus infinity on a log-log plot
    kl_plot = []
    ks_plot = []
    kc_plot = []
    lu_plot = []
    svd_plot = []
    cofac_plot = []

    #Cycles through the lists skipping the 0 points
    for k,lu,svd,c in zip(k_values, lu_err, svd_err, cofactor_err):
        if lu != 0:
            lu_plot.append(lu)
            kl_plot.append(k)
        if svd != 0:
            svd_plot.append(svd)
            ks_plot.append(k)
        if c != 0:
            cofac_plot.append(c)
            kc_plot.append(k)

    #More just plotting stuff
    plt.plot(kc_plot, cofac_plot, color = 'blue', label = "Cofactor error")
    plt.plot(kl_plot, lu_plot, color = 'red', label = "LU error")
    plt.plot(ks_plot, svd_plot, color = 'green', label = "SVD error")

    plt.legend()

    plt.xlabel('Values of k')
    plt.ylabel('Residual squared on solution')
    plt.suptitle('Erros on solution as k approaches 0 on a log-log plot')

    plt.xscale('log')
    plt.yscale('log')

    plt.show()

def plot_t_n_error(n,N):
    #Generates list of times to compute for size 2-7 matrices
    t_2, t_3, t_4, t_5, t_6, t_7 = [],[],[],[],[],[]
    N_list = [2,3,4,5,6,7]
    #Loops through n repeats of completing N inversions. Appending the Times
    # the relevant lists
    for a in range(n):
        t,N_list = get_t_N(8,N)
        t_2.append(t[0])
        t_3.append(t[1])
        t_4.append(t[2])
        t_5.append(t[3])
        t_6.append(t[4])
        t_7.append(t[5])

    #Creates a list of the lists for easier manipulation
    t_list = [t_2,t_3,t_4,t_5,t_6,t_7]

    #return t_list

    #Makes a list of the mean times to complete and the standard deviations
    # on those means
    ave_t_list = [np.mean(time_list) for time_list in t_list]
    std_t_list = [np.std(time_list) for time_list in t_list]

    #Rescales the means so there representive of the mean / (N+1)!
    ave_t_scale = [t/np.math.factorial(n+1) for t,n in zip(ave_t_list,N_list)]
    std_t_scale = [t/(np.sqrt(10)*np.math.factorial(n+1)) for t,n in zip(std_t_list,N_list)]

    #Rescaling so the mean value is at 1, rescaling the errors by the same
    # poporiton
    scaled_ave_t_scale = [t/np.mean(ave_t_scale) for t in ave_t_scale]
    scaled_std_t_scale = [std/np.mean(ave_t_scale) for std in std_t_scale]

    #Normal plotting stuff
    plt.errorbar(range(2,8),scaled_ave_t_scale,scaled_std_t_scale)
    plt.xlabel("Dimension of matrix")
    plt.ylabel("Mean time for 10 inversion / (N+1)!")
    plt.show()

    return scaled_ave_t_scale, scaled_std_t_scale

def manual_chi_squared(time_list, error_list):
    #Removes n = 2 from the data, and then performs a chi squared test for
    # a flat line at a y value equal to the mean
    relevant_times = time_list[1:]
    error_list = error_list[1:]
    expected = np.mean(relevant_times)
    chi_list = [(observed-expected)**2/error for observed,error in zip(relevant_times, error_list)]
    sum_chi = sum(chi_list)
    return sum_chi

def random_walk_max_finder(pitch, starting_point, max_steps, rope_index):
    #Takes random steps around the variable space. Keeps the step if it increases
    # the tension and gets rid of it if Not

    #Starts the camera of at the initial point and finds the tension to begin
    # with
    pitch.camera_position = starting_point
    tension = tension_find(pitch, lu_svd_cofactor = 'lu')
    current_tension = -tension.elements[rope_index]

    #Tries taking the relevant umber of steps, starting with small max_steps
    #which proceed to get smaller.
    for a in range(max_steps):
        x_y = pitch.camera_position
        r = 0.01
        #Second 1800 are 1 big
        if a < 2000:
            r = 1
        #First 200 are 10 big
        if a < 200:
            r = 10
        #Produces a random direction
        theta = rd.uniform(0,2*np.pi)
        #Converting the polar step into cartesian
        delta_x = r*np.cos(theta)
        delta_y = r*np.sin(theta)
        #Taking the step
        new_x_y = tuple(np.add(x_y, (delta_x,delta_y)))
        #Moving the camera to the position
        pitch.camera_position = new_x_y
        #Since tension_find prints and returns nothing if the camera is not camera
        # supported so if the steps off hte supported area, this throws out a
        # AttributeError which is caught and causes the step to reset before
        # continueing the loop.
        try:
            tension = tension_find(pitch, lu_svd_cofactor = 'lu', print = False)
            tension = -tension.elements[rope_index]
        except AttributeError:
            pitch.camera_position = x_y
            continue
        #If the tension is bigger takes the step if not return to the previously
        # step.
        if tension > current_tension:
            x_y = new_x_y
            current_tension = tension
        pitch.camera_position = x_y

    return x_y, current_tension


menu = 'loop'
choice = 'q'
while menu == 'loop':
    print("There are 4 main sections to this demonstration and also instructions \
at the top of the source code on how to use the class and functions. The 4 \
options are 1,2,3,4 and finally q to quit. You can also input graph1 for the first \
graph in the report or table1 and so on.\
\n 1: Option 1 demosntrates the cofactors method of inverting matrices and \
demonstrates this for a 2x2 and 3x3. These techniques are then applied to \
solving randomly generated simultaneous equations. \
\n 2: Option 2 compares this cofactors method to the LU and SVD techniques \
testing their speeds for inverting 4x4s and 5x5s, and then tests them against \
2 nearly singular simultaneous equations 1 with no solutions and 1 with many.\
\n 3: Option 3 explores the tensions involved in suspending a camera above a \
football pitch.\
\n 4: Option 4 finds the point of maximum tension and marks it on the graph.\
\n 5: Option 5 demonstrates tension maps for different cable arangements and\
different sized pitches. ")
    choice = input("Please input which option you wish to be demostrated or q to quit: ")
    if choice == 'q':
        break
    if choice == '1':
        #Generates a random matrix object
        A = random_integer_matrix((2,2))
        #Checks for singularity
        A.get_determinant()
        #Keeps making them until the matrix is non-singular.
        while A.determinant == 0:
            print("Singular matrix generated:\n", A.elements,"\nCreating a new one.\n")
            A = random_integer_matrix((2,2))
            A.get_determinant()
        #Shows the matrix as is
        print("Matrix generated is\n", A.elements)
        input("Hit enter to continue: ")

        A.invert()

        #Shows the inverse
        print("\nThe inverse of this is\n", A.inverse.elements, "\nMultiplied by", A.inverse.scalar)

        input("Hit enter to continue: ")
        print("\nWe can test this my multiplying them together.")

        #Multiplies the matrix and is inverse to make the identity
        C = A.matrix_multiplication(A.inverse)
        C.scalar_in()

        print("\nThe result of this Multiplaction is:\n ",C.elements, "\n which \
        I boldly state is the identity (Sometimes floating point errors result in e-16 factors).")
        input("Hit enter to continue: ")
        print("The same process is quickly shown for a 3x3")

        #Repeats the process for a 3x3
        A = random_integer_matrix((3,3))
        A.get_determinant()

        while A.Get_determinant == 0:
            print("Singular matrix generated created:\n", A.elements,"\nCreating a new one.\n")
            A = random_integer_matrix((3,3))
            A.get_determinant
        A.invert()

        print("Matrix:\n", A.elements)
        print("\nInverse:\n", A.inverse.elements, "\nTimes", A.inverse.scalar)

        C = matrix_multiplication(A, A.inverse)
        C.scalar_in()
        print("\nResult of multiplying the inverse and the matrix is:\n", C.elements)
        input("Please hit enter: ")
    if choice == '2':
        print("First we will generate a random 4x4 matrix and 4 vector to solve.")
        #Makes the random 4x4 and a random 4 vector.
        A = random_integer_matrix((4,4))
        c = random_interger_vector(4)

        #Timing the time for inversion
        t0 = time.time()
        x = solve_simul(A,c)
        t1 = time.time()
        print("\nThe matrix is:\n", A.elements,"\nAnd the vector:\n", c.elements)
        input("Hit enter to continue: ")
        print("The time for cofactor to sovle is: ", round((t1-t0),8))
        print("The solution was found to be:\n", x.elements)

        #Checks the solution for accuracy
        residual = check_solve(A,x,c)
        print("With residual of: ", residual)
        input("Hit enter to continue: ")

        #Repeats the test for the lu and then the svd solutions
        t0 = time.time()
        x = lu_solve(A,c)
        t1 = time.time()
        print("The time for Lu to solve is: ", round((t1-t0),8))
        print("This solution is:\n", x.elements)

        residual = check_solve(A,x,c)
        print("The residual squared: ", residual)
        input("Hit enter to continue: ")

        t0 = time.time()
        x = svd_solve(A,c)
        t1 = time.time()
        print("The time for svd to solve is: ", round((t1-t0),8))
        print("This solution is:\n", x.elements)

        residual = check_solve(A,x,c)
        print("The residual squared: ",residual)
        input("Hit enter to continue: ")

        #Once again repeating the tests but for a 4x4 this time
        print("Just the residuals and times for a 5x5 for each method is presented next")
        A = random_integer_matrix((5,5))
        c = random_float_vector(5)
        input("Hit enter to continue: ")

        t0 = time.time()
        x = solve_simul(A,c)
        t1 = time.time()
        residual = check_solve(A,x,c)
        print("For cofactors the time is: ", round((t1-t0),7), " with residual of: ",residual)
        input("Hit enter to continue: ")

        t0 = time.time()
        x = lu_solve(A,c)
        t1 = time.time()
        residual = check_solve(A,x,c)
        print("For LU the time is: ", round((t1-t0),7), "with residual of: ",residual)
        input("Hit enter to continue: ")

        t0 = time.time()
        x = svd_solve(A,c)
        t1 = time.time()
        residual = check_solve(A,x,c)
        print("For SVD the time is: ", round((t1-t0),7), "with residual of: ",residual)
        input("Hit enter to continue: ")

        print("Next shown is the residuals on solving the nearly multi-solutioned simultaneous equations")
        input("Hit enter to continue: ")
        plot_near_singular(multi_no = 'multi')
        print("Now will be the residuals on solving the nearly no solutioned simultaneous equations")
        plot_near_singular(multi_no = 'no')
    if choice == '3':
        print("This section explores the tension in the ropes used to suspend the camera.")
        input("Please hit enter to continue: ")
        print("This plot has the tensions for the rope in the bottom left.")
        #Generates the deafult pitch and then plots the tensions map
        pitch = Pitch()
        plot_tension_map(pitch, 0, 100, 70)
        print("The tnesion for bottom right are just symettric to this so have been ommited")
        input("Please hit enter to continue: ")

        print("This plot has the tensions for the rope based at the top of the triangle")
        plot_tension_map(pitch, 2, 100, 70)

        input("Please hit enter to continue: ")
        print("This plot is for the sum of the tensions in all of the ropes")
        plot_tension_map(pitch, 3, 100, 70)

        input("Please hit enter to continue: ")
    if choice == '4':
        #Crates a pitch object
        pitch = Pitch()
        #Finds maximum tension value and position
        point, tension = random_walk_max_finder(pitch, (40,40), 10000, 0)
        #Plots a tension heat map with that point marked
        plot_tension_map(pitch, 0, 100, 70, point = point)
        print("The maximum tension occurs at: ",round(point[0],3), round(point[1],3)," with a tension of", round(tension,3))
        input("Please hit enter to continue: ")
    if choice == '5':
        print("A right angled triangle tension map maximum marked. Maps printed in order bottom left, bottom right, top, total")
        #Making a pitch and moving the cable ends
        choice_5_pitch = Pitch()
        choice_5_pitch.cable_ends = [(5,-5), (95,-5), (5,75)]
        #Plots for each cable tension and total tension
        for i in [0,1,2,3]:
            point = (-50,-50)
            if i != 3:
                point, t = random_walk_max_finder(choice_5_pitch, (10,10), 10000, i)
            plot_tension_map(choice_5_pitch, i, 100,70, point = point)
            if i != 3:
                print("The maximum occurs at ",point," with tension ", t, "N")

        input("Please hit enter to continue: ")
        print("A larger pitch with same triangle roughly maximum marked")
        #Creates a pitch and changes its dimensions and cable end position
        choice_5_pitch_new = Pitch()
        choice_5_pitch_new.length = 130
        choice_5_pitch_new.width = 100
        choice_5_pitch_new.cable_ends = [(5,-5), (125,-5), (65,105)]
        #Plots for each cable tension and total tension with the maximum tension
        # marked for the indivual cables
        for i in [0,1,2]:
            point,t = random_walk_max_finder(choice_5_pitch_new, (10,2), 10000, i)
            plot_tension_map(choice_5_pitch_new, i, 100, 70, point = point)
            print("The maximum occurs at ",point," with tension ", t, "N")
        plot_tension_map(choice_5_pitch_new, 3, 100, 70)
        input("Please hit enter to continue: ")

    if choice == 'graph1':
        input("This is a smaller quicker version of the plot, going to a lower\
        matrix size with less iterations")

        #Generates a list of time taken versus N
        t,N = get_t_N(7,3)
        #Normalises t by 1000t/(N+1)!
        t_normalised = [1000*a/np.math.factorial(b+1) for a,b in zip(t,N)]

        #Normal plotting information
        plt.plot(N, t_normalised)
        plt.xlabel("Size of matrix")
        plt.ylabel("Time taken x 1000 / (N+1)!")
        plt.suptitle("Testing the (N+1)! complexiety")
        #plt.savefig("T N Relationship.jpg")
        plt.show()
    if choice in ['graph2', 'graph3']:
        #Both graphs are just generated in this function
        plot_near_singular(multi_no = 'multi')
    if choice == 'graph4':
        #Graph is just generated within this plot
        plot_near_singular(multi_no = 'no')
    if choice == 'graph5':
        #Plots the tension map for the first rope in the basic set up
        graph5_Pitch = Pitch()
        plot_tension_map(graph5_Pitch, 0, 100, 70)
    if choice == 'graph6':
        #Does the same for graph 5 but seocnd rope
        graph6_Pitch = Pitch()
        plot_tension_map(graph6_Pitch, 1, 100, 70)
    if choice == 'graph7':
        #Does the same for graph 5 and 6 but for the third rope
        graph7_Pitch = Pitch()
        plot_tension_map(graph7_Pitch, 2, 100, 70)
    if choice == 'graph8':
        #Does teh same for graph 5,6 and 7 hut for the total tension
        graph8_Pitch = Pitch()
        plot_tension_map(graph8_Pitch, 2, 100, 70)
    if choice == 'table1':
        #Generates the times for inverting 5 3x3 a 4x4 and 5x5
        data = make_dict([3,4,5], [5,5,5])
        #Places this data into a pandas data framme
        Table = pd.DataFrame(data)
        #Reads out the columns as lists
        cofactor = Table[Table.columns[1]]
        LU = Table[Table.columns[2]]
        SVD = Table[Table.columns[4]]
        repeats = Table[Table.columns[3]]
        cofactor_scaled = [30*time/repeat for time,repeat in zip(cofactor, repeats)]
        LU_scaled = [30*time/repeat for time,repeat in zip(LU, repeats)]
        SVD_scaled = [30*time/repeat for time,repeat in zip(SVD, repeats)]
        Table['C scaled'] = cofactor_scaled
        Table['Scaled LU'] = LU_scaled
        Table['Scaled SVD'] = SVD_scaled
        Table
