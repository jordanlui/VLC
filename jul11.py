# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:08:15 2017

@author: Jordan

Looking at data results carefully in code and plots
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from TDMS_handler import runTDMS
import os.path

#%% Functions
def normpower(xx):
    for i in range(3,7):
        col = xx[:,i]
        newcol = (col - np.min(col)) / (np.max(col) - np.min(col))
        xx[:,i] = newcol
    return xx

def make_contour(x,y,z,interp_method='nearest',levels=5,boolplot=0):
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
#%% Main
dir_in = '../Data/july14/'
file = "10khz_1.tdms"
#path_TDMS = dir_in + file + '.tdms'
path_in = dir_in + 'analysis/' + file[:-5] + '.csv'
if os.path.isfile(dir_in+file):
    print 'file exists. Will not run TDMS Importer'
else:
    
    runTDMS(dir_in,file) # Run TDMS Import script
    
interp_method = 'nearest' # Interpolation method used for griddata function
boolplot = 0
# Load data
x_all=[]
#x_all.append(np.genfromtxt('../Data/july14/analysis/1khz_circleslow_1.csv',delimiter=','))
#x_all.append(np.genfromtxt('../Data/july14/analysis/1khz_circles_3.csv',delimiter=','))
#x_all.append(np.genfromtxt('../Data/july14/analysis/1khz_circles_2.csv',delimiter=','))
x_all.append(np.genfromtxt(path_in,delimiter=','))
# Combine with others
xx = np.vstack(x_all)
m = len(xx)
segment = 0.2
#np.random.shuffle(xx)
#xx = xx[0:int(segment*m),:]
#xx = xx[::250]

# Normalize our power data
maxpower = np.max(xx[:,3:])
minpower = np.min(xx[:,3:])
#xx = normpower(xx)
#xx[:,3:7] = np.round(xx[:,3:7],decimals=4)


time = xx[:,0] # Extract time value
x = xx[:,1]
y = xx[:,2]
c1 = xx[:,3]
c2 = xx[:,4]
c3 = xx[:,5]
c4 = xx[:,6]

c1 = c2 # A real dumb switching method. but effective for now

#%% Try a moving average
window_width=500
cumsum_vec = np.cumsum(np.insert(c1, 0, 0)) 
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
len_new = len(ma_vec)
# Make a new coord, time vector to match. Note we should check and understand how the window is working
x = x[:len_new]
y = y[:len_new]
time = time[:len_new]
c1 = ma_vec



#%% Data handling
#x = np.round(x,decimals=1)
#y = np.round(y,decimals=1)

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
plt.plot(time[slice1:slice2],c1[slice1:slice2],"o")
plt.title("Signal power with time")

plt.figure(3)
plt.plot(y[slice1:slice2],c1[slice1:slice2],"o")
plt.title("Signal power with distance")


#%% Contour Plot and Surface plot
# Surface plot is probably more applicable
#f2, (ax21, ax22) = plt.subplots(1,2)

#make_contour(x,y,c1,interp_method=interp_method,levels=5)
make_contour(x,y,c1,interp_method=interp_method,levels=5,boolplot=boolplot)
#make_contour(x,y,c3,interp_method=interp_method,levels=5)
#make_contour(x,y,c4,interp_method=interp_method,levels=5)
#fig = plt.figure()
##ax1 = fig.add_subplot(111,projection='3d')
#
## Interpolation script based on the inputs
#xmin = int(np.min(x))
#xmax = int(np.max(x))
#ymin = int(np.min(y))
#ymax = int(np.max(y))
#z=c1
#npts = len(x)
#shift = 0 # Shift amount for the linspace
#xi = np.linspace(xmin-shift, xmax+shift,100)
#yi = np.linspace(ymin-shift, ymax+shift,100)
#zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='nearest')
## contour the gridded data, plotting dots at the randomly spaced data points.
#CS = plt.contour(xi,yi,zi,5,linewidths=0.5,colors='k')
#CS = plt.contourf(xi,yi,zi,5,cmap=plt.cm.jet)
#plt.title('contour plot')
#plt.colorbar() # draw colorbar
## plot data points.
#plt.scatter(x,y,marker='o',c='b',s=5)
#plt.xlim(xmin,xmax)
#plt.ylim(ymin,ymax)
#plt.title('Contour map of measurements (%d points)' % npts)
#plt.show()

#%% Mesh Grid
#x2,y2 = np.meshgrid(xi,yi)
#
#fig = plt.figure(3)
#ax = fig.add_subplot(111, projection='3d')
#ax1.plot_surface(x2,y2,zi,cmap=cm.coolwarm,linewidth=0)

# Scatter plotting
# Show up quite messy. Hard to interpret
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x,y,c1,c='blue')
#ax.scatter(x,y,c2,c='white')
#ax.scatter(x,y,c3,c='green')
#ax.scatter(x,y,c4,c='red')