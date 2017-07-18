# -*- coding: utf-8 -*-
"""
Created on Sun Jul 09 20:36:23 2017

@author: Jordan
"""
import numpy as np
import random
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#from scipy import asarray as ar,exp
import scipy.fftpack

def dataprep(x,seed,segment,twidth):
    # Takes our x matrix and shuffles, segments, and extracts label values
    # Takes t values based on twidth. Assumes labels start in column 0.
    # Shuffle data based on the random seed input
    random.seed(a=seed)
    x = np.asarray(random.sample(x,len(x)))
    
    # Find index at which we segment our data    
    segindex = int(len(x)*segment)
    
    # Segment our data (May29 data set formatting rules)
    
    # Grab X values. 
    #twidth = 2 # We have two columns of label data in front of x matrix from May 29
    x_train = x[:segindex,twidth:]
    x_test = x[segindex:,twidth:]
    
    # Segment T values 
    t_train = x[:segindex,:twidth]
    t_test = x[segindex:,:twidth]
    
    # Reshape t to proper array if twidth was 1, so we are not dealing with list
    if twidth == 1:
        t_train = np.reshape(t_train,(t_train.shape[0],twidth))
        t_test = np.reshape(t_test,(t_test.shape[0],twidth))

    return x_train,x_test,t_train,t_test
    
def unison_shuffled_copies(a, b):
    # Shuffles to arrays together in unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def stats(f):
    # Stats on data
    # Current format: min,max,mean,stdev,stdev/mean
    stats=[]
    for i in range(0,f.shape[1]):
        col = f[:,i]
        
        min = np.min(col)
        max = np.max(col)
        mean = np.mean(col)
        stdev = np.std(col)
        stdevrel = stdev/mean
        print min,max,mean,stdev
        stats.append(np.array((min,max,mean,stdev,stdevrel)))
    stats = np.asarray(stats)
    return stats

def makeFFT(y):
    # Make an FFT and plot

    # Number of samplepoints
    N = len(y)
    # sample spacing - does this correspond to frequency at all?
    T = 1.0 / 800.0
#    x = np.linspace(0.0, N*T, N)
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    fig, ax = plt.subplots()
    ax.plot(xf, 10 * np.log10(2.0/N * np.abs(yf[:N//2])))
    plt.title('FFT, y axis log scale')
    plt.show()
    return xf,yf

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 
    
def calcSma(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
#    empty_list = [None] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(sub_result)

def smoothingPlot(t,x,N1=50,N2=50):
    # Smooth data and plot it for comparison
    f, axarr = plt.subplots(3, sharex = True)
    axarr[0].plot(t,x,'x')
    axarr[0].set_title('Data vs time, no smoothing')
    
    
    # Try smoothing and compare
#    N1 = 100 # Number of running average points
    x_cum = running_mean(x,N1)
    axarr[1].plot(t[:len(x_cum)], x_cum, 'x')
    title = 'Data vs time, Cumulative average on %d points' %N1
    axarr[1].set_title(title)
    
#    N2 = 500
    x_sma = calcSma(x,N2)
    #c1_smooth2 = np.asarray(c1_smooth2)
    axarr[2].plot(t[:len(x_sma)], x_sma, 'x')
    title = 'Data vs time, Simple moving average over %d points' %N2
    axarr[2].set_title(title)
    
    print 'data mean ', np.mean(x),'relative stdev before is ',np.std(x)/np.mean(x), 'stdev cum avg is', np.std(x_cum)/np.mean(x_cum), 'and SMA stdev is ', np.std(x_sma)/np.mean(x_sma)
    return x_cum, x_sma