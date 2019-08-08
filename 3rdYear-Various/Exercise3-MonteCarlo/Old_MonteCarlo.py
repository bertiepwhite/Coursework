import os
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import time
import random

#---------------------------------Part 1-------------------------------------#
def bin_list(list_input,max,bins):
    bin_end = np.linspace(max/bins,max,bins)
    print(bin_end)

    bin_counts = []
    list_input.sort()

    for end in bin_end:
        for i,value in enumerate(list_input):
            if value > end:
                bin_counts.append(i)
                break
    bin_counts.append(len(list_input))

    bin_counts = np.array(bin_counts)

    bin_counts[1:] = bin_counts[1:] - bin_counts[:-1]

    return bin_counts

##################
def sin_random_analaytical():
    gen = np.random.uniform(low = -1, high = 1)
    return mt.acos(gen)

###################
def sin_random_acc_rej():
    gen = np.random.uniform(low = 0, high = mt.pi)
    test_lim = mt.sin(gen)
    test_val = np.random.random()
    if test_val < test_lim:
        return gen
    else:
        gen = sin_random_acc_rej()
        return gen

####################
def lorentz_acc_rej(speed=0):
    beta = speed/3e8
    gen = np.random.uniform(low = -pi/2, high = pi/2)
    test_lim = (mt.cos(gen) + beta)/(1+beta*mt.cos(gen))
    test_val = np.random.random()
    if test_val < test_lim:
        return gen
    else:
        gen = lorentz_acc_rej(speed = speed)
        return gen

#---------------------------------Part 2-------------------------------------#

def decay_disance(mean_life, speed):
    #Mean life in micro_seconds
    lorentz = 1/np.sqrt(1-(speed/3e8)**2)
    life_time = np.random.exponential(mean_life*lorentz)
    return life_time * 2000 * 10**(-6)

def board_position(mean_life, speed, total_distance, blurred = True, angles=False):
    #All Non-physical results are ignored
    gamma_distance = total_distance - decay_disance(mean_life,speed)
    while gamma_distance < 0 or gamma_distance > total_distance:
        gamma_distance = total_distance - decay_disance(mean_life,speed)
    #Change to whats best
    x_angle = lorentz_acc_rej(speed = speed)
    y_angle = lorentz_acc_rej(speed = speed)
    #We need to sample the whole circle so will randomly add a pi factor

    x_pos = gamma_distance * mt.tan(x_angle)
    y_pos = gamma_distance * mt.tan(y_angle)

    if blurred:
        x_blur = np.random.normal(scale = 0.1)
        y_blur = np.random.normal(scale = 0.3)

        x_pos += x_blur
        y_pos += y_blur

    if angles:
        return x_pos,y_pos,x_angle,y_angle

    return x_pos, y_pos

def detector_spread(points,mean_life,speed,total_distance,returns=False):
    x_list = []
    y_list = []
    for a in range(points):
        x,y = board_position(mean_life,speed,total_distance)
        x_list.append(x)
        y_list.append(y)

    plt.scatter(x_list,y_list)
    plt.xlim((-5,5))
    plt.ylim((-6,6))
    plt.show()
    if returns:
        return zip(x_list,y_list)

def bin_2d(data, size = (2.5,3),pixel_size = (0.1,0.3)):

    x_bins = mt.ceil((size[0]*2)/pixel_size[0])
    y_bins = mt.ceil((size[1]*2)/pixel_size[1])
    detector = np.zeros(shape = (x_bins,y_bins))
    for point in data:
        x_cord = point[0]+(size[0])
        y_cord = -point[1]+(size[1])
        if  0 < x_cord < (size[0]*2) and 0 < y_cord < (size[1]*2):
            i = mt.floor(x_cord/pixel_size[0])
            j = mt.floor(y_cord/pixel_size[1])
            detector[i][j] += 1

    #In order to fix an intuitive x and y on plt.imshow()
    return detector.transpose()

def solid_angle(width,height,dist):
    alpha = np.abs(width/(2*height))
    beta = np.abs(width/(2*height))
    upper = 1+alpha**2+beta**2
    lower = (1+alpha**2)*(1+beta**2)
    return 4*mt.acos(np.sqrt(upper/lower))

def solid_angle_pixel(x,y,size,d):
    positives = solid_angle(2*x+size, 2*y+size,d) + solid_angle(2*x-size,2*y-size,d)
    negatives = solid_angle(2*x+size, 2*y-size,d) + solid_angle(2*x-size,2*y+size,d)
    return np.abs((positives-negatives)/4)

def analytical(size,pixels,d):
    x_values = np.linspace(-size,size,pixels)
    y_values = np.linspace(-size,size,pixels)
    detector = np.zeros((pixels,pixels))
    pixel_size = 2*size/pixels
    for i,x in enumerate(x_values):
        for j,y in enumerate(y_values):
            detector[i][j] = solid_angle_pixel(x,y,pixel_size,d)
    return detector

def angle_distribution(points):
    x_angles = []
    y_angles = []
    for a in range(points):
        x_p,y_p,x_a,y_a = board_position(520,2000,2,angles = True)
        x_angles.append(x_a)
        y_angles.append(y_a)

    r = [1] * len(x_angles)
    x_bins = bin_list(x_angles,np.pi,20)
    y_bins = bin_list(y_angles,np.pi,20)
    y_angles = [a+np.pi for a in y_angles]

    ax = plt.subplot(111,projection = 'polar')
    ax.scatter(x_angles,r,c='r',label = 'theta')
    ax.scatter(y_angles,r,c='b',label = 'phi')
    ax.legend()
    ax.set_rticks([0.5,1,1.5])
    ax.set_rmax(1.5)
    plt.show()

    theta = np.linspace(0,np.pi,20)
    phi  = np.linspace(np.pi,2*np.pi,20)

    ax = plt.subplot(111,projection = 'polar')
    ax.plot(theta,x_bins,c = 'r', label = 'theta')
    ax.plot(phi,y_bins,c = 'b', label = 'phi')
    ax.legend()
    plt.show()

#----------------------------Part 3-------------------------#

def background(mean,error):
    gen_mean = np.random.normal(loc=mean,scale = error)
    return np.random.poisson(lam = gen_mean)

def count_sim(L,sigma,back_mean,back_error):
    back_count = background(back_mean,back_error)
    gen_count = np.random.poisson(lam = (L*sigma))
    return (back_count + gen_count)

def count_rates(L,sigma,back_mean,back_error,repeats):
    count_numbers = np.zeros(int(L*sigma*5))
    for a in range(repeats):
        count = count_sim(L,sigma,back_mean,back_error)
        count_numbers[count] += 1
    back_trim = np.trim_zeros(count_numbers,'b')
    full_trim = np.trim_zeros(back_trim)
    start = len(back_trim) - len(full_trim)
    return full_trim, start

def sigma_search(sigma_min,sigma_max,L,back_mean,back_error):
    sigmas = np.linspace(sigmaa_min,sigma_max,20)

def speed_accuracy_test(iterations, repeats):

    accept_reject = []
    analytically = []
    accept_reject_times = []
    analytical_times = []

    for j in range(repeats):
        t0 = time.time()
        for i in range(iterations):
            accept_reject.append(sin_random_acc_rej())
        t1 = time.time()

        accept_reject_times.append(t1-t0)

        t0 = time.time()
        for i in range(iterations):
            analytically.append(sin_random_analaytical())
        t1 = time.time()

        analytical_times.append(t1-t0)

    accept_reject_time = np.mean(accept_reject_times)
    analytical_time = np.mean(analytical_times)

    accept_reject_err = np.std(accept_reject_times)/(np.sqrt(repeats-1))
    analytical_err = np.std(analytical_times)/(np.sqrt(repeats-1))

    acc_height, bin_edges, patches = plt.hist(accept_reject, bins = 100,
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

    acc_chi = np.sum((acc_height-expected)**2/acc_error)
    ana_chi = np.sum((ana_height-expected)**2/ana_error)

    print("Accept reject method produces a chi square value of ", acc_chi)
    print("Analytical method procudes a chi square value of ", ana_chi)

    plt.errorbar(mid_points, acc_height, yerr = acc_error, color = 'r', label = "Accept Reject")
    plt.errorbar(mid_points, ana_height, yerr = ana_error, color = 'b', label = "Analytical")
    plt.plot(mid_points, expected, color = 'g')
    plt.legend()
    plt.show()



    print("Accept reject took: ", accept_reject_time, " with an error of: ",
           accept_reject_err, "\nAnalytical took: ", analytical_time,
           " with an error of: ", analytical_err)
