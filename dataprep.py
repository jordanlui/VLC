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
from scipy.interpolate import griddata

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

def running_mean_old(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def running_mean(x, N):
    if len(x.shape) > 1 : # Check if x is a vector or matrix. If matrix, x.shape should return a len 2 tuple
        # If x is not just a vector we need to complete this column wise
        newx = []
        for i in range(0,x.shape[1]):
            col = x[:,i]
            cumsum = np.cumsum(np.insert(col, 0, 0)) 
            col =  (cumsum[N:] - cumsum[:-N]) / N
#            col = np.reshape(col, (len(col),1))
#            newx = np.hstack((newx,col))
            newx.append(col)
        x = np.asarray(newx).transpose()
    else:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        x  = (cumsum[N:] - cumsum[:-N]) / N
    return x
    
def calcSma(data, smaPeriod):
    # Simple Moving Average (SMA) calculation
    # Built to accept vectors or matrices
    # This SMA script appears to "eat from the front" of the dataset, as opposed to the end

    if len(data.shape) > 1: # Check if we have a MxN matrix input
#        print 'matrix input detected'
        newx = []
        x = data # save our matrix into new object
        for i in range(0,data.shape[1]): # iterate through the columns
            data = x[:,] # Grab the column
            j = next(i for i, x in enumerate(data) if x is not None)
            our_range = range(len(data))[j + smaPeriod - 1:]
            sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]
            newx.append(sub_result) # Append result
        sub_result = np.asarray(newx).transpose()
        
    else:

        j = next(i for i, x in enumerate(data) if x is not None)
        our_range = range(len(data))[j + smaPeriod - 1:]
    #    empty_list = [None] * (j + smaPeriod - 1)
        sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(sub_result)
def SMA_orig(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
    empty_list = [None] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(empty_list + sub_result)
    return 0
def smoothingPlot(t,x,N1=50,N2=50):
    # Smooth data and plot it for comparison
    # Note the SMA "eats from the front" of the dataset
    f, axarr = plt.subplots(3, sharex = True)
    axarr[0].plot(t,x,'x')
    axarr[0].set_title('Data vs time, no smoothing')
    
    # Try smoothing and compare
    # Cumulative aka running mean
    x_cum = running_mean(x,N1)
    ind_start = int((N1-1)/2)
    ind_end = len(t) - (N1-ind_start) + 1
#    print len(t), len(x_cum), ind_start, ind_end, len(t[ind_start:ind_end])
    axarr[1].plot(t[ind_start:ind_end], x_cum, 'x') # Running average "eats data from both ends"
    title = 'Data vs time, Cumulative average on %d points' %N1
    axarr[1].set_title(title)

    # Smooth Moving Average
    x_sma = calcSma(x,N2)
    axarr[2].plot(t[N2-1:], x_sma, 'x') # Note how we reduce t value, since SMA "eats from front" of array
    title = 'Data vs time, Simple moving average over %d points' %N2
    axarr[2].set_title(title)
    
    print 'data mean ', np.mean(x),'relative stdev before is ',np.std(x)/np.mean(x), 'stdev cum avg is', np.std(x_cum)/np.mean(x_cum), 'and SMA stdev is ', np.std(x_sma)/np.mean(x_sma)
    return x_cum, x_sma
def make_contour(x,y,z,interp_method='nearest',levels=5,boolplot=0,file='null'):
    # Make contour plot from arrays of data. Not a lin space and interpolation is performed
    fig = plt.figure()
    #ax1 = fig.add_subplot(111,projection='3d')
    
    # Interpolation script based on the inputs
    xmin = int(np.min(x))
    xmax = int(np.max(x))
    ymin = int(np.min(y))
    ymax = int(np.max(y))
    npts = len(x)
    shift = 0 # Shift amount for the linspace
    xi = np.linspace(xmin-shift, xmax+shift,100)
    yi = np.linspace(ymin-shift, ymax+shift,100)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method=interp_method)
    # contour the gridded data, plotting dots at the randomly spaced data points.
    plt.contour(xi,yi,zi,levels,linewidths=0.5,colors='k')
    plt.contourf(xi,yi,zi,levels,cmap=plt.cm.jet)
    title = 'contour ' + file[:-5] + ' ' + interp_method
    plt.title(title)
    
    plt.colorbar() # draw colorbar
    # plot data points.
    if boolplot:
        plt.scatter(x,y,marker='o',c='b',s=5)
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        title = title + ' Contour map (%d points)' % npts
    plt.title(title)
    return fig