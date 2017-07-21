# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:25:02 2017

@author: Jordan
Try a Gaussian Fit
"""
import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import *
from scipy import optimize
from scipy import stats
import os
import glob
from scipy.interpolate import griddata

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
def example():
    Xin, Yin = np.mgrid[0:201, 0:201] # Meshgrid
    data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + np.random.random(Xin.shape)
    
    plt.matshow(data, cmap=plt.cm.gist_earth_r)
    
    params = fitgaussian(data)
    fit = gaussian(*params)
    
    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params
    
    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f""" %(x, y, width_x, width_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    return 0
def jGaussian(x,y,z,interp_method='nearest'):
    # Fit a 2D Gaussian
    # Inputs are all arrays
    xmin = int(np.min(x))
    xmax = int(np.max(x))
    ymin = int(np.min(y))
    ymax = int(np.max(y))
    npts = len(x)
    shift = 0 # Shift amount for the linspace
    xi = np.linspace(xmin-shift, xmax+shift,100)
    yi = np.linspace(ymin-shift, ymax+shift,100)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method=interp_method) # Generate our interpolated grid data
    data = zi
    
    plt.matshow(data, cmap=plt.cm.gist_earth_r)
    
    params = fitgaussian(data)
    fit = gaussian(*params)
    
    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params
    
    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f""" %(x, y, width_x, width_y),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    
    
    return params

#%% Example Code
# Main Script to fit 2D Gaussian
#http://scipy-cookbook.readthedocs.io/items/FittingData.html
#example()

#%% Try this again with our data


# Parameters
# Input file directory
#filename = 'circle_slow_1.csv'
#dir_in = "../Analysis/june27/"
#path_in = dir_in + filename
## Output file parameters
#dir_out = '../analysis/june27/'
#fileout = filename[:-5] + '.csv' # output file name of the averages file, named based on input file. Clipped to remove TDMS filename
#path_out = os.path.join(dir_out,fileout)
#
#xx = np.genfromtxt(path_in,delimiter=',')
#t = xx[:,3]
#x = xx[:,1]
#y = xx[:,2]