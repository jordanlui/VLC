# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 22:58:05 2017

@author: Jordan
data grid test
"""
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
# make up some randomly distributed data
seed(1234)
npts = 200
x = uniform(-2,2,npts) # The irregular data values
y = uniform(-2,2,npts)
z = x*np.exp(-x**2-y**2) # z function based on their input
# define grid.
xi = np.linspace(-2.1,2.1,100) #The lin space
yi = np.linspace(-2.1,2.1,100)
# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic') # Grid data function accepts irregular data as well as the linspace
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
#plt.scatter(x,y,marker='o',c='b',s=5)
#plt.xlim(-2,2)
#plt.ylim(-2,2)
#plt.title('griddata test (%d points)' % npts)
#plt.show()
