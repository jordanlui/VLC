# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 13:12:47 2017

@author: Jordan

# July 14 analysis

# Test 1 Translation
File names list the distances in cm
"""

from __future__ import division
import glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from TDMS_handler import tdmsfuncjun
from natsort import natsorted
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import scipy.fftpack




# Parameters
path = '../Data/july14/'
pathout = path  + 'analysis/'
summaryfile = 'summary.csv'
filelist = glob.glob(path+'translationtest1_pos*.tdms') # Filelist
filelist = natsorted(filelist) # Sort in logical order to make thinking easier

# Functions
def fitGaussian(x,y):
    # Some preformatting based on the current july 14 input data
    x = np.array(map(float,x))
    y = np.array(y)

    
    n = float(len(x))                          #the number of data
    mean = float(np.sqrt(sum(x*y)/n ) )                 #note this correction
    sigma = float(sum(y*(x-mean)**2)/n )       #note this correction
    a = np.max(y)          # Guess the Gaussian Amplitude based on largest value
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    
    popt,pcov = curve_fit(gaus,x,y,p0=[a,mean,sigma])
    ypred = gaus(x,*popt)
    SST = np.sum( (y-np.mean(y))**2)
    SSR = np.sum( (y-ypred)**2 )
    R2 = 1 - SSR/SST
    
    
    plt.figure()
    plt.plot(x,y,'b+:',label='data')
    plt.plot(x,ypred,'ro-',label='fit')
    plt.legend(bbox_to_anchor=(1, 1),
               bbox_transform=plt.gcf().transFigure)
    plt.title('Guassian fit')
    plt.xlabel('Distance')
    plt.ylabel('Signal power')
    plt.show()
    return popt, ypred, R2


# Empty arrays

c1=[]
c2=[]
c3=[]
c4=[]
dist = []
x=[]
y=[]
c1var=[]

# Process all files and take stats on signals
def loadFiles(filelist):
    for file in filelist:
        f = tdmsfuncjun(file) # Load TDMS file to load our data. Format [t,x,y,c1,c2,c3,c4]
    #    print f.shape
        c1.append(np.mean(f[:,3]))
        c2.append(np.mean(f[:,4]))
        c3.append(np.mean(f[:,5]))
        c4.append(np.mean(f[:,6]))
        x.append(np.mean(f[:,1]))
        y.append(np.mean(f[:,2]))
        c1var.append(np.var(f[:,3]))
        
        dist_search = re.search('translationtest1_pos_(.+)\.tdms',file,re.IGNORECASE)
        if dist_search:
            adist =  dist_search.group(1)
            dist.append(adist)
            
    summary = np.array((x,y,dist,c1,c2,c3,c4)).transpose()
    with open(os.path.join(pathout,summaryfile),'wb') as file:
        np.savetxt(file,summary,fmt='%s',delimiter=',')
    return 0
       
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
def smoothing_window(x,N):
    # Performs simple moving window smoothing
    return 0
    
def calcSma(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
    empty_list = [None] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(sub_result)


# Ended
        
#%% Main


        
#%% Examine individual Static data files
path = '../Data/july14/static/analysis/'
file = '1khz_static_1.csv'
f = np.genfromtxt(path+file,delimiter=',')

# Slice the data to take a close look
start = 0
end = 2e3
#f = f[start:end,:]

# Stat analysis on static (non moving) tests
t,x,y,c1,c2,c3,c4 = zip(*f)
t,x,y,c1,c2,c3,c4 = np.array((t,x,y,c1,c2,c3,c4))

stats = stats(f) # Compute stats (min,ma,mean,stdev,etc) on data
# max deviation on c1
maxdev = np.max(np.abs(c1 - np.mean(c1)))
print 'max deviation from mean is', maxdev

f, axarr = plt.subplots(3, sharex = True)
axarr[0].plot(t,c1,'x')
#plt.plot(t,c2,'o')
axarr[0].set_title('Data vs time, no smoothing')


# Try smoothing and compare
N = 100 # Number of running average points
c1_smooth = running_mean(c1,N)
axarr[1].plot(t[:len(c1_smooth)], c1_smooth, 'x')
title = 'Data vs time, Cumulative average on %d points' %N
axarr[1].set_title(title)

N = 100
c1_smooth2 = calcSma(c1,N)
#c1_smooth2 = np.asarray(c1_smooth2)
axarr[2].plot(t[:len(c1_smooth2)], c1_smooth2, 'x')
title = 'Data vs time, Simple moving average over %d points' %N
axarr[2].set_title(title)

print 'data mean ', np.mean(c1),'relative stdev before is ',np.std(c1)/np.mean(c1), 'stdev cum avg is', np.std(c1_smooth)/np.mean(c1_smooth), 'and SMA stdev is ', np.std(c1_smooth2)/np.mean(c1_smooth2)

# Fourier Analysis
y=c4
xf,yf = makeFFT(y)
#%% Split and look
xf2=xf[0:10]
yf2=yf[0:10]
N = len(y)
plt.figure()
plt.plot(xf2, yf2 )
plt.title('FFT on subset of data')



#%% Fit a Gaussian on the movement data
#loadFiles(filelist)
#popt, ypred, R2 = fitGaussian(dist,c3) # fit gaussian, output parameters, and plot
#print popt, R2
#%% Plot results
#plt.figure()
#plt.plot(dist,c1,'o')
#plt.title('1khz signal')        
#        
#plt.figure()
#plt.plot(dist,c1,'o')
#plt.plot(dist,c2,'x')
#plt.plot(dist,c3,'o')
#plt.plot(dist,c4,'x')
#plt.title('all signals, non-normalized')
