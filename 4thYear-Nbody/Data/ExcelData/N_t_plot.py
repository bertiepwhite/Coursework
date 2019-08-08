import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import linalg as la
import scipy.fftpack

font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 22}

matplotlib.rc('font', **font)


def SVD(matrix, constants):
    U, sigma, Vh = la.svd(matrix)

    leng = len(matrix)
    sigma = la.diagsvd(sigma, leng, leng)

    U_t = np.transpose(U)
    V = np.transpose(Vh)

    for i in range(len(sigma)):
        if sigma[i][i] != 0:
            sigma[i][i] = 1/(sigma[i][i])

    sigma = np.transpose(sigma)

    x = np.matmul(V, sigma)
    x = np.matmul(x, U_t)
    x = np.matmul(x, constants)

    return x

def N_t_plot():
    def func(x,m,c):
        return m*x+c

    file = pd.read_csv("Varying_N_t.csv")
    N_t = np.array(file['Time steps'])
    Par4 = np.array(file['Time Taken (4)'])[:-2:]
    Par14 = np.array(file['Time Taken (14)'])[:-2:]
    GPU = np.array(file['Time Taken (GPU)'])
    Lin = np.array(file['Time Taken (Lin)'])

    plt.plot(np.log(N_t[:-2]),np.log(Par4), color = 'r',linewidth = 4, label = 'OMP (4)')
    plt.plot(np.log(N_t[:-2]),np.log(Par14), color = 'b',linewidth = 4, label = 'OMP (14)')
    plt.plot(np.log(N_t),np.log(GPU),color = 'g',linewidth = 4, label = 'GPU')
    plt.plot(np.log(N_t),np.log(Lin),color = 'k',linewidth = 4, label = 'Lin')
    plt.legend()
    plt.ylabel("log(Time taken) (log(s))")
    plt.xlabel("log(Number of Steps)")
    plt.show()

def N_plot():
    file = pd.read_csv("Vary_N.csv")
    N = np.array(file['Particle Number (Steps=2000)'])
    GPU = np.array(file['Time  (GPU)'])
    Lin = np.array(file['Linear'])[:-3]
    Par14 = np.array(file['Time (P14)'])[:-2]
    Par4 = np.array(file['Time (P4)'])[:-2]

    def func(x,m,c):
        return m*x+c

    GpuPopt, a = curve_fit(func,np.log(N[-6:]),np.log(GPU[-6:]))
    GpuPopt2,a = curve_fit(func,np.log(N[:4]),np.log(GPU[:4]))
    LinPopt, a = curve_fit(func,np.log(N[-8:-3]),np.log(Lin[-5:]))
    ParPopt, a = curve_fit(func,np.log(N[-7:-2]),np.log(Par4[-5:]))
    ParPopt2, a = curve_fit(func,np.log(N[-7:-2]),np.log(Par14[-5:]))

    print("GPU: ", GpuPopt,"\nGPU FLAT REGION: ", GpuPopt2)
    print("LIN: ", LinPopt)
    print("PAR4: ", ParPopt)
    print("PAR14: ", ParPopt2)

    Gpux = np.log(N)
    Lynx = np.log(N[:-3])
    Parx = np.log(N[:-2])

    plt.scatter(Gpux,np.log(GPU),color='g',label='GPU')
    plt.scatter(Lynx,np.log(Lin),color='k',label='Lin')
    plt.scatter(Parx,np.log(Par4),color='r',label='OMP (4)')
    plt.scatter(Parx,np.log(Par14),color='b',label='OMP (14)')

    plt.plot(Gpux[-12:],func(Gpux[-12:],GpuPopt[0],GpuPopt[1]),color='g')
    plt.plot(Gpux[:6]  ,func(Gpux[:6]  ,GpuPopt2[0],GpuPopt2[1]),color='g')
    plt.plot(Lynx[-10:],func(Lynx[-10:],LinPopt[0],LinPopt[1]),color='k')
    plt.plot(Parx[-12:],func(Parx[-12:],ParPopt[0],ParPopt[1]),color='r')
    plt.plot(Parx[-12:],func(Parx[-12:],ParPopt2[0],ParPopt2[1]),color='b')

    plt.xlabel("log(Number of particles)")
    plt.ylabel("log(Time), log(s)")

    plt.legend()
    plt.show()

def thread_plot():

    file = pd.read_csv("Vary_threads.csv")
    threads = np.array(file['Threads'])
    threads2 = np.arange(15000).astype(float)
    threads2 = threads2/1000 + 1
    T4000 = np.array(file['Time (4000)'])
    T2000 = np.array(file['Time (2000)'])
    T1000 = np.array(file['Time (1000)'])
    T500 = np.array(file['Time (500)'])
    T200 = np.array(file['Time (40000)'])

    def func4000(x,Fl,T1):
        return (T1*(Fl+(1-Fl)/x))
    def func2000(x,Fl,T1):
        return (T1*(Fl+(1-Fl)/x))
    def func1000(x,Fl,T1):
        return (T1*(Fl+(1-Fl)/x))
    def func500(x,Fl,T1):
        return (T1*(Fl+(1-Fl)/x))
    def func200(x,Fl,T1):
        return (T1*(Fl+(1-Fl)/x))


    T4000popt, a = curve_fit(func4000,threads,T4000)
    T2000popt, a = curve_fit(func2000,threads,T2000)
    T1000popt, a = curve_fit(func1000,threads,T1000)
    T500popt, a = curve_fit(func500,threads,T500)
    T200popt, a = curve_fit(func200,threads,T200)

    print("T4000: ",T4000popt)
    print("T2000: ",T2000popt)
    print("T1000: ",T1000popt)
    print("T500: ",T500popt)
    print("T40000: ",T200popt)

    plt.scatter(threads,2/T4000,color = 'k')
    plt.scatter(threads,2/T1000,color = 'g')
    plt.scatter(threads,1/T2000,color = 'r')
    plt.scatter(threads,1/T500,color = 'b')
    #plt.scatter(threads,1/T200,color= 'y')

    #plt.plot(threads2, 1/func200(threads2,T200popt[0],T200popt[1]),color = 'y',label = '40000 Particles')
    plt.plot(threads2, 2/func4000(threads2,T4000popt[0],T4000popt[1]),color = 'k',label = '4000 Particles',linewidth=4)
    plt.plot(threads2, 2/func1000(threads2,T1000popt[0],T1000popt[1]),color = 'g',label = '1000 Particles',linewidth=4)
    plt.plot(threads2, 1/func2000(threads2,T2000popt[0],T2000popt[1]),color = 'r',label = '2000 Particles',linewidth=4)
    plt.plot(threads2, 1/func500(threads2,T500popt[0],T500popt[1]),color = 'b',label = '500 Particles',linewidth=4)
    #plt.ylim((0.9*np.min(1/T200)-0.00005,1.1*np.max(1/T200)+0.000005))
    #plt.xlim((0,17))
    plt.xlabel("Thread Number")
    plt.ylabel("Time 1/s")

    plt.legend()
    plt.show()

def GPU_low_N():
    file = pd.read_csv("GPU_low_N.csv")
    GPU_N = np.array(file['Particle Number (Steps=2000)'])
    Times = np.array(file['Time  (GPU)'])


    NBoolless = GPU_N<900
    NBoolmore = GPU_N>1100

    low_N = GPU_N[NBoolless]
    big_N = GPU_N[NBoolmore]

    low_time = Times[NBoolless]
    big_time = Times[NBoolmore]

    def func(x,m,c):
        return m*x+c

    lowpopt, a = curve_fit(func,low_N,low_time)
    bigpopt, a = curve_fit(func,big_N,big_time)

    plt.scatter(np.log(GPU_N),np.log(Times))
    plt.show()

def GPU_N(big=False, small=True):
    file = pd.read_csv("GPU_N.csv")

    def func(x,m,c):
        return m*x+c

    N = np.array(file[file.keys()[0]])
    T = np.array(file[file.keys()[1]])
    LINN = np.array(file[file.keys()[2]][3:7])
    TIMEN = np.array(file[file.keys()[3]][3:7])
    if small:
        N_small = N < 1000
        popt, a = curve_fit(func,np.log(N[N_small]),np.log(T[N_small]))
        print(popt)
        plt.plot(np.log(N[N_small]),func(np.log(N[N_small]),popt[0],popt[1]),color = 'r',linewidth=4)
        plt.scatter(np.log(N[N_small]),np.log(T[N_small]),s=42,color='k')

    if big:
        N_big = N >= 1000
        popt, a = curve_fit(func,np.log(N[N_big]),np.log(T[N_big]))
        print(popt)
        plt.plot(np.log(N[N_big]),func(np.log(N[N_big]),popt[0],popt[1]),color='b',linewidth=4)
        plt.scatter(np.log(N[N_big]),np.log(T[N_big]),s=42,color='k')
    popt, a = curve_fit(func,np.log(LINN),np.log(TIMEN/5))
    plt.plot(np.log(LINN),func(np.log(LINN),popt[0],popt[1]),color='g',linewidth=4,label='Linear')
    plt.scatter(np.log(LINN),np.log(TIMEN/5),s=42,color='g')
    plt.xlabel("log(Number of Particles)")
    plt.ylabel("log(Time) log(s)")
    plt.legend()
    plt.show()

def FFT_energy():
    Energy = np.genfromtxt("energies.csv")
    Energy = Energy[Energy.shape[0]//20::] # Cropping the first 5%
    Energy -= np.mean(Energy)

    N = Energy.shape[0]
    T = 1/(24*60)
    freqs = np.fft.fftfreq(Energy.shape[0])
    FFT = np.fft.fft(Energy)
    amps = np.abs(FFT)

    x = np.linspace(0.0, N*T, N)
    Ef = scipy.fftpack.fft(Energy)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    plt.plot(xf, 2.0/N * np.abs(Ef[:N//2]))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()


def CrossOverFind():
    file = pd.read_csv("CrossOver.csv")
    N_lin = np.array(file['Lin N'])[:20]
    T_lin = np.array(file['Time Lin'])[:20]

    Par4N = np.array(file['Par 4N'])[3:]
    Par4T = np.array(file['Time 4'])[3:]

    ParN = np.array(file['ParN'])
    P14T = np.array(file['Time 14'])
    P18T = np.array(file['Par 18'])

    plt.plot(N_lin,T_lin,linewidth=4,color= 'k',label = 'Linear')
    plt.plot(Par4N,Par4T,linewidth=4,color = 'r',label = 'Par (4)')
    plt.plot(ParN,P14T,linewidth=4,color = 'b',label = 'Par(14)')
    plt.plot(ParN,P18T,linewidth=4,color = 'g',label = 'Par(18)')
    plt.legend()
    plt.xlabel("Number of Particles")
    plt.ylabel("Time (s)")
    plt.show()

def GPU_Chunk():
    file = pd.read_csv("GPU_CHUNKS.csv")


    CS2000 = np.array(file[file.keys()[0]])[:-1]
    TT2000 = np.array(file[file.keys()[1]])[:-1]
    CS16384 = np.array(file[file.keys()[2]])
    TT16384 = np.array(file[file.keys()[3]])
    plt.plot(np.log(CS2000),np.log(TT2000))
    plt.plot(np.log(CS16384),np.log(TT16384))
    plt.show()
