# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:08:15 2017

@author: Jordan

Looking at data results carefully in code and plots

Examining moving data and generating contour plots
Note data smoothing required
Analysis on July 11 data
Making Contour plots
Analysis on July 19 data, creating contour plots to verify transmitter location


"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from TDMS_handler import runTDMS
import os.path
from dataprep import make_contour, calcSma
import glob


#%% Main
dir_in = '../Data/july19/static/analysis/'

interp_method = 'nearest' # Interpolation method used for griddata function
boolplot = 1 # Flag for plotting contour points or not

# Load data
x_all=[]
filelist = glob.glob(dir_in + '*.csv') # Generate file list
filelist = filelist[:4]
for file in filelist:
    
    x_all.append(np.genfromtxt(file,delimiter=','))
# Combine with others
xx = np.vstack(x_all)
m = len(xx)

#Segment and shuffle as required
#segment = 0.2
#np.random.shuffle(xx)
#xx = xx[0:int(segment*m),:]
#xx = xx[::250]

# Normalize our power data
maxpower = np.max(xx[:,3:])
minpower = np.min(xx[:,3:])
#xx = normpower(xx)
#xx[:,3:7] = np.round(xx[:,3:7],decimals=4)

# Extract values
time = xx[:,0] # Extract time value
x = xx[:,1]
y = xx[:,2]
c1 = xx[:,3]
c2 = xx[:,4]
c3 = xx[:,5]
c4 = xx[:,6]

channel = c2 # A real dumb switching method. but effective for now
#channel = np.resize(channel,(len(channel),1))
#%% Data treatment and smoothing
# Change logic order since we should data smoothing on individual files
N1 = 300
smooth_channel = calcSma(channel,N1)
smooth_x = x[N1-1:]
smooth_y = y[N1-1:]
# Data handling
x = np.round(x,decimals=1)
y = np.round(y,decimals=1)

#%% Plotting

# Coordinate plot
plt.figure(1)
plt.plot(time,y,"o")
plt.plot(time,x,"x")
plt.title('x and y coordinate changes with time')

# Look at some data
slice1=0
slice2=int(m*0.3)
plt.figure(2)
plt.plot(time[slice1:slice2],channel[slice1:slice2],"o")
plt.title("Signal power with time")

plt.figure(3)
plt.plot(y[slice1:slice2],channel[slice1:slice2],"o")
plt.title("Signal power with distance")



#%% Contour Plot and Surface plot
# Surface plot is probably more applicable
#f2, (ax21, ax22) = plt.subplots(1,2)


make_contour(x,y,channel,interp_method=interp_method,levels=15,boolplot=boolplot,file=file)

# Look at the smooth data
make_contour(smooth_x,smooth_y,smooth_channel,interp_method=interp_method,levels=15,boolplot=boolplot,file=file)

