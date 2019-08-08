import os
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import time
import random
from matplotlib import gridspec
from scipy.special import gammainc
from scipy.stats import linregress
from scipy.optimize import curve_fit


#######################    Part 1 function   #############################

def accept_reject():
    #An accept reject method for producing a sinusodial random number

    #Producing a random number within the domain of sine
    gen = np.random.uniform(low = 0, high = mt.pi)

    #Applying sine to find the P(gen) desired and then producing a random number
    #between 0 and 1 for it to pass/fail
    test_lim = mt.sin(gen)
    test_val = np.random.random()

    #Returning the number if it passed or recursively retrying until a pass is
    # found
    if test_val < test_lim:
        return gen
    else:
        gen = accept_reject()
        return gen

def analaytical():
    #Since the integral of sin is cos and cos has a range of -1 to 1 the domain
    # of acos we wannt is the same.
    gen = np.random.uniform(low = -1, high = 1)
    return mt.acos(gen)

def speed_accuracy_test(iterations,repeats):
    #Timing a total of iterations*repeats angles broken into 'repeats' number
    # of blocks

    #Emtpy lists for the data to be stored in
    acc_rej = []
    analytically = []
    accept_reject_times = []
    analytical_times = []

    #Timing how long 'iterations' number of angles takes to generate and repeating
    # this repeat times
    for j in range(repeats):

        #Timing accept reject storing all the anlges
        t0 = time.time()
        for i in range(iterations):
            acc_rej.append(accept_reject())
        t1 = time.time()

        #storing the times
        accept_reject_times.append(t1-t0)

        #timing and storing the same for the analytical method
        t0 = time.time()
        for i in range(iterations):
            analytically.append(analaytical())
        t1 = time.time()
        analytical_times.append(t1-t0)

    #Finding average time and standard error
    accept_reject_time = np.mean(accept_reject_times)
    analytical_time = np.mean(analytical_times)

    accept_reject_err = np.std(accept_reject_times)/(np.sqrt(repeats-1))
    analytical_err = np.std(analytical_times)/(np.sqrt(repeats-1))

    #Using plt.hist to bin our data into 100 bins. Patches is not used
    acc_height, bin_edges, patches = plt.hist(acc_rej, bins = 100,
                                                  range = (0,np.pi))
    ana_height, bin_edges, patches = plt.hist(analytically,  bins = 100,
                                                  range = (0,np.pi))
    #Clearing the plot
    plt.clf()

    #The mid points and expected are equal for both cases
    mid_points = (bin_edges[1:] + bin_edges[:-1])*0.5
    expected = np.sin(mid_points)

    #Using counting statistics to calculate the error
    acc_error = np.sqrt(acc_height)
    ana_error = np.sqrt(ana_height)

    #Calculating the area of the histogram, since each plot adds an area of
    # bin width * (change bin height) which is 1
    bin_width = bin_edges[1] - bin_edges[0]
    hist_area = bin_width * repeats * iterations

    #Scaling the bins so that the total area is the same as that for a sin
    # between 0 and pi
    acc_height *= (2/hist_area)
    ana_height *= (2/hist_area)
    acc_error  *= (2/hist_area)
    ana_error  *= (2/hist_area)

    #Scaling the bins
    acc_chi = np.sum((acc_height-expected)**2/acc_error)
    ana_chi = np.sum((ana_height-expected)**2/ana_error)

    print("Accept reject method produces a chi square value of ", acc_chi)
    print("Analytical method procudes a chi square value of ", ana_chi)

    sin_height = np.sin(mid_points)

    plt.plot(mid_points,sin_height, color = 'g', label = 'True sine')
    plt.errorbar(mid_points, acc_height, yerr = acc_error, color = 'r', label = "Accept Reject")
    plt.errorbar(mid_points, ana_height, yerr = ana_error, color = 'b', label = "Analytical")
    plt.plot(mid_points, expected, color = 'g')
    plt.legend()
    plt.show()



    print("Accept reject took: ", accept_reject_time, " with an error of: ",
           accept_reject_err, "\nAnalytical took: ", analytical_time,
           " with an error of: ", analytical_err)

################       Part 2 and 3 Classes     ########################

class Detector():
    """
    A class that allows the simulation of a detector array of any given size and
    of given pixel size able to account for relitivistic beam speeds.
    """
    def __init__(self, size = (3,3.6), pixel_size = (0.1,0.3)):
        #Storing and Calculating useful information about the geometery of the
        #detector.

        #Information given to us
        self.size = size
        self.pixel_size = pixel_size

        #Pixel resolution of the detector
        self.x_pixels = mt.ceil((size[0]*2)/pixel_size[0])
        self.y_pixels = mt.ceil((size[1]*2)/pixel_size[1])

        #Storing the borders of the pixels useful in many areas
        self.x_edges = np.linspace(-size[0],size[0], self.x_pixels + 1)
        self.y_edges = np.linspace(-size[1],size[1], self.y_pixels + 1)

        self.x_err = 0.1
        self.y_err = 0.3

        #Preloaded fit gathered from large samples can be inputted for this
        self.A_fit = 13839360.6
        self.c_fit = 65.4782876

        #Stroing the data for the true position of the gamma ray intercepts with
        # the detector aswell as the pixel detector
        self.points = []
        self.detector = np.zeros(shape = (self.y_pixels,self.x_pixels))

    def gauss_func(self,x,sig,mu,A,c):
        return (A/np.sqrt(2*np.pi*sig**2))*(np.exp((x-mu)/sig**2))+c

    def quick_run(self, mean_life,speed,total_distance,points):
        #Short A quick combination of all the functions required to reach a
        # detector image
        self.set_beam(mean_life,speed,total_distance)
        self.beam_fire(points)
        self.blur_points()
        self.detector_count()

    def lorentz_angle(self):
        #A random angle generator that follows the lorentz transformed cos

        #Since the only angles we require are ones that interect with the plane
        #the detector is on
        gen = np.random.uniform(low = -np.pi/2, high = np.pi/2)

        #Setting the test lim aka L(gen) with L as the distribution
        test_lim =  (mt.cos(gen) + self.beta)/(1+self.beta*mt.cos(gen))
        test_val = np.random.random()

        #The same return/repeat system as sine in accept_reject()
        if test_val < test_lim:
            return gen
        else:
            gen = self.lorentz_angle()
            return gen

    def set_beam(self, mean_life, speed, total_distance):
        #Setting the values relating to the beam

        #Mean life in micro_seconds
        self.mean_life = mean_life
        self.speed = speed
        self.total_distance = total_distance

        #Finding the relitivistic variables
        self.beta = speed/3e8
        self.lorentz = 1/(np.sqrt((1-self.beta**2)))

    def beam_fire(self, points):
        #A function to find the detector positions of "points" number of gamma
        # rays

        #Ensuring a baem has been prepared
        if 'speed' not in self.__dict__:
            print("Beam has not been prepared, use set_beam to prepare beam")
            return

        #Repeating the process the right number of times
        for a in range(points):

            #Finding a random decay time following the exponential
            life = np.random.exponential(self.mean_life*self.lorentz)

            #Calculating distance left after decay, if the decay occured after
            #the detector just continuing onto the next particle
            dist_remain = self.total_distance - (life*self.speed*(10**-6))
            if dist_remain < 0:
                continue

            #Finding the 2 angles
            x_angle = self.lorentz_angle()
            y_angle = self.lorentz_angle()

            #Finding the board positions
            x_pos = dist_remain * mt.tan(x_angle)
            y_pos = dist_remain * mt.tan(y_angle)

            #Adding the points to the list
            self.points.append((x_pos,y_pos))

    def blur_points(self):
        #Adding a gaussian smear to all the points
        if len(self.points) == 0:
            print("No points to blur")
            return

        #storing the new blurred points
        self.blurred = []
        for point in self.points:
            x_blur = (point[0]+np.random.normal(scale = self.x_err))
            y_blur = (point[1]+np.random.normal(scale = self.y_err))
            self.blurred.append((x_blur,y_blur))

    def detector_count(self):
        #Counting all the points within each pixel and storing it as an array

        #Ensuring theres points to blur
        if 'blurred' not in self.__dict__:
            print("There must be blurred points available to bin")

        for point in self.blurred:
            #Changing origin from the centre of detector to the top left
            x_cord =  point[0] + self.size[0]
            y_cord = -point[1] + self.size[1]

            #Discarding data if outside the detector and finding the pixel the
            # decay point lies within
            if  0 < x_cord < (self.size[0]*2) and 0 < y_cord < (self.size[1]*2):
                j = mt.floor(x_cord/self.pixel_size[0])
                i = mt.floor(y_cord/self.pixel_size[1])
                self.detector[i][j] += 1

    def half_count_square(self):
        #Finding the rectangle centred on the origin containing half the radius

        #Ensuring the detector isnt empty
        if self.detector.sum() == 0:
            print("Empty detector")
            return

        #finding half the points and the middle value for the x and y
        test_lim = len(self.points)/2
        mid_x = self.x_pixels/2
        mid_y = self.y_pixels/2

        #Finding the initial x and y slice values, containing the 2 central
        # pixels if the middle of x is inbetween 2 pixels
        if self.x_pixels % 2 == 1:
            low_x  = int(mid_x)
            high_x = low_x + 1
        else:
            high_x = round(mid_x) + 1
            low_x  = high_x - 2

        #Repeating teh process for y
        if self.y_pixels % 2 == 1:
            low_y  = int(mid_y)
            high_y = low_x + 1
        else:
            high_y = round(mid_y) + 1
            low_y  = high_y - 2

        #The initial test value
        old_val = 0
        test_val = self.detector[low_y:high_y][:,low_x:high_x].sum()

        #Expanding out the square until over half the points are enclosed
        while test_val < test_lim:
            #Storing the old values for area and pixels contained for extrop
            old_area = (high_x-low_x-1)*(high_y-low_y-1)
            old_val = test_val
            #Expanding the square
            low_x  -= 1
            low_y  -= 1
            high_x += 1
            high_y += 1
            #If the square is outside the detector finish and report
            if high_y > len(self.detector):
                print("The detector missed over half the points run with a\
                larger detector")
            #Update the test value
            test_val = self.detector[low_y:high_y][:,low_x:high_x].sum()

        #Extrapolating between pixels to find a more exact
        area = (high_x-low_x-1)*(high_y-low_y-1)
        ratio = (test_val - old_val)/(area - old_area)
        diff_val = test_val - test_lim
        diff_area = diff_val/ratio

        #Storing the area and working out the calculated mean life
        self.half_area_bound = [low_x,high_x,low_y,high_y]
        self.half_area  = old_area - diff_area
        self.predicted_mean_life = (self.A_fit/(self.speed*self.half_area**2)
                                   + self.c_fit)

    def detector_plot(self):
        #Plotting the detector
        extent = [-self.size[0],self.size[0],-self.size[1],self.size[1]]
        plt.imshow(self.detector,extent = extent, origin = 'upper')
        plt.colorbar()
        plt.show()

    def full_plot(self):

        #Scaling the column sums so the total area is ~5 then shifting to be on
        # the y axis
        x_sums = (sum(self.detector)*10
                 /sum(sum(self.detector))
                 -self.size[1])
        y_sums = (sum(self.detector.transpose())*3
                 /sum(sum(self.detector))
                 -self.size[0])

        #Creating a linspace tatt hits the middle of every column and a second
        #for rows
        x_cords = np.linspace(-self.size[0]+self.pixel_size[0]/2,
                               self.size[0]-self.pixel_size[0]/2,
                               num = len(x_sums))
        y_cords = np.linspace(-self.size[1]+self.pixel_size[1]/2,
                               self.size[1]-self.pixel_size[1]/2,
                               num = len(y_sums))


        #Plotting the scaled and shifted column sums
        plt.scatter(x_cords,x_sums,color = 'r',label = 'Column Sums')
        #Plotting the scaled and shifted row sums
        plt.scatter(y_sums,y_cords,color = 'w', label = 'Row Sums')

        #Plotting the detector and extra info
        extent = [-self.size[0],self.size[0],-self.size[1],self.size[1]]
        plt.imshow(self.detector,extent = extent, origin = 'upper')
        plt.colorbar()
        plt.legend(loc = 'upper right')
        plt.show()

    #########################################################################
    #############    All of these form the analytic solution   ##############
    #########################################################################

    def solid_angle(self,width,height):
        #Solid angle of a rectangle centred on the origin of a given width and
        # height at the same place as the detector

        #Variables which are simpler to wortk with
        alpha = np.abs(width/(2*self.total_distance))
        beta = np.abs(height/(2*self.total_distance))

        #The denominator and numerator within the square root
        upper = 1+alpha**2+beta**2
        lower = (1+alpha**2)*(1+beta**2)

        #etturning the solid angle
        return 4*mt.acos(np.sqrt(upper/lower))

    def solid_angle_pixel(self,x,y):
        #Solid angle on a detector a distance d away at position x,y

        #Solid angle of a rectangle centred on the origin to the top right of the
        # pixel
        O1 = self.solid_angle((2*(x+self.pixel_size[0])),(2*(y+self.pixel_size[1])))
        #Solid angle of a rectangle centred on the origin to the bottom left of
        # the pixel
        O2 = self.solid_angle((2*x),(2*y))
        #The 2 rectangles that carve out the centre of O1 without the pixels
        #included
        O3 = self.solid_angle((2*(x+self.pixel_size[0])),(2*y))
        O4 = self.solid_angle((2*x),(2*(y+self.pixel_size[1])))

        return np.abs((O1+O2-O3-O4))/4

    def analytic_detector_solve(self):
        #Creating a blank detector and inputting the solid angle of each pixel
        # this equals the probablity assuming all decays happen from
        self.analytic_detector = np.zeros_like(self.detector)
        for j,x in enumerate(self.x_edges[:-1]):
            for i,y in enumerate(self.y_edges[1:]):
                self.analytic_detector[i][j] = self.solid_angle_pixel(np.abs(x)
                                                                     ,np.abs(y))

    def analytic_plot(self):
        #Plotting the analytic version of the detector
        extent = [-self.size[0],self.size[0],-self.size[1],self.size[1]]
        plt.imshow(self.analytic_detector,extent = extent)
        plt.colorbar()
        plt.show()

class Distributions():
    """
    A method to simulate a background plus some event with an event with a cross
    section and Luminosity. And determining the likelihood of this distribution
    from an anomolous background.
    """
    def __init__(self, mean_background, err_background, lum_mean, lum_std):
        #Preserving the initial data with errors for both the background and
        # lumonisity consindered.
        self.back_mean = mean_background
        self.back_err  = err_background
        self.lum = lum_mean
        self.lum_err = lum_std

    def sim_background(self):
        #Generating a background rate by first finding a gaussian random mean
        gen_mean = np.random.normal(loc = self.back_mean, scale = self.back_err)
        return np.random.poisson(lam = gen_mean)

    def count_sim(self, L, sigma):
        #simulating a background and a random
        background = self.sim_background()
        gen_count = np.random.poisson(lam = (L*sigma))
        return background + gen_count

    def sim_distribution(self,sigma,repeats,distributions):
        #Generate an empty list to count the number simulations with the count
        # rate equal to the index. Need to be high enough that a simmed count rate
        # is not larger than the
        count_rates = np.zeros((distributions,(int((self.back_mean+self.lum*sigma)*10))))
        for i in range(distributions):
            for a in range(repeats):
                L = np.random.normal(loc = self.lum, scale = self.lum_err)
                count = self.count_sim(L,sigma)
                count_rates[i][count] += 1

        sum_count_rates = sum(count_rates)

        #Discarding all zeros at the end of the array
        back_trim = np.trim_zeros(sum_count_rates,'b')
        #Discarding all zeros at the front of the array
        full_trim = np.trim_zeros(back_trim)
        #Finding what count the first number on the array corrrosponds to
        start = len(back_trim) - len(full_trim)

        mean_count_rates = full_trim / distributions

        err = np.array([np.std(col) for col in count_rates.transpose()])

        self.start = start
        self.distribution  = mean_count_rates
        self.error = err[:len(self.distribution)]
        self.end = start + len(self.distribution)

    def background_distribution(self):
        mid_points = np.linspace(self.start,self.end-1,
                                 num = len(self.distribution))
        mean = np.random.normal(loc = self.back_mean, scale = self.back_err)
        poisson = [np.exp(-mean)*mean**k/mt.factorial(k) for k in mid_points]
        total_points = np.sum(self.distribution)
        self.back_poisson = total_points * np.array(poisson)

    def chi_square_calc(self):
        self.background_distribution()
        chi_squared = 0
        for obs,exp,err in zip(self.distribution,self.back_poisson,self.error):
            if err != 0:
                chi_squared += ((obs-exp)**2)/(err**2)

        self.chi_squared = chi_squared
        #We're fitting a distrivution with no paramters
        degrees_freedom = self.end-self.start
        self.p_value = gammainc(degrees_freedom/2,chi_squared/2)

    def candidate_events(self):
        offset = 5-self.start
        if offset < 0:
            offset = 0
        self.percent_candidate = sum(self.distribution[offset:])/sum(self.distribution)

    def distribution_plot(self):
        self.background_distribution()
        mid_points = np.linspace(self.start,self.end-1,
                                 num = len(self.distribution))

        plt.bar(mid_points,self.back_poisson,width = 1,fill = False,
                edgecolor = 'b', linewidth = 3, label = 'Background')
        plt.bar(mid_points,self.distribution,yerr = self.error,
                width = 1,fill = False, edgecolor = 'r', linewidth = 3,
                label = 'Simulated Data')
        plt.ylabel("Count Rates")
        plt.xlabel("Counts")
        ymin, ymax = plt.ylim()
        plt.ylim((0,ymax))
        plt.legend()
        plt.xlim((self.start-0.5,self.end+0.5))
        plt.show()


##################Fucntions using the classees written##################

def calibrate_mean_life_spread(full_short = 'full'):
    print("This is slower unless full_short = 'short' has been passed")
    #Setting the different half lifes to be ran depending on wether its just
    # an example or an actual fit being done
    if full_short == 'short':
        half_lifes = [220,320,420,520,620]
    if full_short == 'full':
        half_lifes = np.linspace(10,1000,250)

    #Finding and storing the relevant spread information
    spreads = []
    for half_life in half_lifes:
        print(half_life)
        #Using a larger detector than usually to avoid missing points
        Det = Detector(size = (100,120))
        Det.quick_run(half_life,2000,2,100000)
        Det.half_count_square()
        spreads.append(Det.half_area)

    return spreads,half_lifes

def fit_inverse_square(x_cords,y_cords,fit_repeats):
    #Fitting a y = A/x**2 + c by first fitting the first and last point then
    # random walking over paramter space

    #Enforcing the 2 systems to be arrays
    x_cords = np.array(x_cords)
    y_cords = np.array(y_cords)
    #Finding first and last point
    first_x, first_y = x_cords[0] , y_cords[0]
    last_x , last_y  = x_cords[-1], y_cords[-1]

    #Finding initial guess
    x_factor = 1/(first_x**2) - 1/(last_x**2)
    A = (first_y-last_y)/x_factor
    c = first_y - A/(first_x**2)
    y_fit = A/(x_cords**2) + c

    #Looping through minimizing residual squared by only changing 1 variable at
    # a time and alternating which is changed
    for a in range(fit_repeats):

        #Working out the changes to c required to fit the indivual data point
        # then averaging them and applying that shift
        c_shift = np.mean(y_cords - y_fit)
        c += c_shift
        y_fit = A/(x_cords**2) + c

        #Repeating the same processes but weighting each c_shift by the residual
        # squared as this is what is desired to be minimised
        c_shifts = y_cords - y_fit
        residuals = (y_cords - y_fit)**2
        weighted_shifts = c_shifts*residuals
        c_shift = np.sum(weighted_shifts) / (len(residuals)*np.sum(residuals))
        c += c_shift

        #Recalculating the fit for the new c
        y_fit = A/(x_cords**2) + c

        #Calculating the cahnges in A required to fit indivual data poitns and
        # finding the mean of them then shifting A by this meaned quality
        A_shift = np.mean((y_fit-y_cords)*x_cords**2)
        A += A_shift
        y_fit = A/(x_cords**2) + c

        #Repeating the same normal then residual fit processes
        A_shifts = (y_fit-y_cords)*x_cords**2
        residuals = (y_cords - y_fit)**2
        weighted_shifts = A_shifts * residuals
        A_shift = np.sum(weighted_shifts) /  (len(residuals)*np.sum(residuals))
        A += A_shift

        #Calculating new sums
        y_fit = A/(x_cords**2) + c


        #Calculating the residual square sum and if the changes to A and c were
        # close to the floating error repeatdly ending the loop
        residual_sum = np.sum((y_fit-y_cords)**2)
        print("Current Residual Square: ", residual_sum)
        if np.abs(A_shift) < 10**(-13) and np.abs < 10**(-13):
            count += 1
            if count == 5:
                break
            else:
                count == 0

    #forming the title for the plot with the A,c calculted in the title
    title = "A = {:.9g} ,  c = {:.9g}".format(A,c)

    #Plotting the fit and the data
    plt.plot(x_cords,y_fit,label = 'fit', color = 'r')
    plt.plot(x_cords,y_cords,label = 'data', color = 'b')
    plt.xlabel("Spread")
    plt.ylabel("Mean Life")
    plt.title(title)
    plt.legend()
    plt.show()

def test_inverse_square(mean_life):
    #Finding the difference between predicted and actual mean lifes

    #Making unnecessarily big
    Det = Detector(size = (10,12))
    #Plotting 10,000 points rather than the usual 100,000 for significant time
    #savings then finding half count square
    Det.quick_run(mean_life, 2000,2,10000)
    Det.half_count_square()

    #Returning the diffrence between predicted and observed
    return Det.predicted_mean_life - mean_life

def inverse_square_test():
    #Generating a random half life then testing its predicted and mean life
    #values

    #Producing a mean life then testing it to see how the predicticted value
    # does
    differences = np.zeros(100)
    for i in range(100):
        mean_life = np.random.uniform(low = 20, high = 1200)
        differences[i] = test_inverse_square(mean_life)

    #Just using the absolute of teh differences
    differences = np.absolute(differences)
    differences.sort()
    plt.plot(differences)
    plt.xlabel("Index")
    plt.ylabel("Differences")
    plt.show()

    stand_dev = differences[68]
    print("The error was found to be ", stand_dev)

def chi_sigma_find(short = False):
    sigmas = np.linspace(0.1, 0.3, num = 41)
    repeats = 250
    repeats2 = 100
    if short:
        sigmas = np.linspace(0.1,0.3, num = 21)
        repeats = 50
        repeats2 = 50
    p_values = []
    for sigma in sigmas:
        p_values_repeats = []
        for j in range(repeats):
            Dist = Distributions(5.8,0.4,10,0.25)
            Dist.sim_distribution(sigma,repeats2,repeats2)
            Dist.chi_square_calc()
            p_values_repeats.append(Dist.p_value)
        p_values.append(np.mean(p_values_repeats))

    return sigmas,p_values

def sig_sigma_find(short = False):

    sigmas = np.linspace(0.01, 0.80, num = 80)
    if short:
        sigmas = np.linspace(0.01,0.80, num = 25)
    p_values = []
    for sigma in sigmas:
        p_values_repeats = []
        for j in range(10):
            Dist = Distributions(5.8,0.4,10,0.25)
            Dist.sim_distribution(sigma,100,100)
            Dist.candidate_events()
            p_values_repeats.append(Dist.percent_candidate)
        p_values.append(np.mean(p_values_repeats))

    return sigmas,p_values

def fit_logistic_manual(x,y):
    x,y = np.array(x),np.array(y)
    x1,y1 = x[0],y[0]
    x2,y2 = x[-1],y[-1]
    #grabbing a central index for a third point
    mid = int(len(x)/2)
    x3,y3 = x[mid],y[mid]

    i_saved = 'Empty'
    for i,y_val in enumerate(y):
        if y_val > 0.5:
            break

    y_diff = y[i] - y[i-1]
    x_diff = x[i] - x[i-1]
    grad = y_diff/x_diff
    y_extra = 0.5 - y[i-1]
    x_extra = y_extra/grad
    x_c = x_extra + x[i-1]

    inital_index = int(len(x)/5)

    grad = linregress(x[:inital_index],y[:inital_index])[0]
    k = grad*4

    y_fit = 1/(1+np.exp(-k*(x-x_c)))
    plt.plot(x,y_fit, color = 'r', label = 'Fit',linewidth = 4)
    plt.plot(x,y, color = 'b', label = 'Data')
    plt.xlabel("Cross section (nb)")
    plt.ylabel("Significance")
    plt.plot(range(2),[0.95]*2, '--')
    plt.xlim((x[0],x[-1]))
    plt.legend(loc = 'lower right')
    plt.show()

    significant = -np.log(1/0.95-1)/k + x_c
    print("The cross over into significance occurs at: ", significant)

def fit_logistic_auto(x,y):
    params,conv = curve_fit(sigmoid_func,x,y)

    k,x_c = params

    y_fit = 1/(1+np.exp(-k*(x-x_c)))
    plt.plot(x,y_fit, color = 'r', label = 'Fit',linewidth = 4)
    plt.plot(x,y, color = 'b', label = 'Data')
    plt.xlabel("Cross section (nb)")
    plt.ylabel("Significance")
    plt.plot(range(2),[0.95]*2, '--')
    plt.xlim((x[0],x[-1]))
    plt.legend(loc = 'lower right')
    plt.show()

    significant = -np.log(1/0.95-1)/k + x_c
    print("The cross over into significance occurs at: ", significant)

def sigmoid_func(x,k,x_c):
    return 1/(1+np.exp(-k*(x-x_c)))


menu = 'loop'
while menu != 'q':
    menu = input("Please input 1,2 or 3 to see part 1 2 or 3  or q to quit: ")
    if menu == '1':
        input("Please hit enter to see how the speed and chi square values \
for the 2 different ways to generate angles: ")

        speed_accuracy_test(1000,100)

    if menu == '2':
        input("Please hit enter to see a variety of detectors: ")

        print("A standard non-relitivistic detector")
        Det = Detector()
        Det.quick_run(520,2000,2,10000)
        Det.detector_plot()
        Det.half_count_square()
        print("The half count rectangle area is: ", Det.half_area)

        print("A relativistic detector")
        Det = Detector()
        Det.quick_run(0.01,1.04e8,2,10000)
        Det.detector_plot()
        Det.half_count_square()
        print("The half count rectangle area is: ", Det.half_area)

        input("Please hit enter to see an example fit to a cross secion chi plot: ")
        print("Warning these plots use criminally low data so get a weak relationship\
a more detailed plot can be (and has been ran) to get good A and c values\
and a print statement for each half life currently on will be printed")

        spreads, half_lifes = calibrate_mean_life_spread(full_short='short')

        fit_inverse_square(spreads,half_lifes,10)

    if menu == '3':
        input("Please hit enter to see a series of generated distributions for\
a background plus some production rates: ")
        Dist = Distributions(5.8,0.4,10,0.25)
        Dist.sim_distribution(0.34,100,100)
        Dist.chi_square_calc()
        Dist.distribution_plot()
        print("The chi squared value calculated is: ", Dist.chi_squared)

        input("Please hit enter to see the finding of the critical values: ")
        print("Warning this can take a while")
        s,p = chi_sigma_find(short = True)
        fit_logistic_manual(s,p)

        s,p = sig_sigma_find(short = True)
        fit_logistic_auto(s,p)
