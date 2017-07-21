# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:38:34 2017

@author: Jordan
gaussian making and other investigations

Try 2D Gaussian on our static recorded data from July 19

"""
from __future__ import division
from gaussian import fitgaussian, moments, gaussian, jGaussian, example
from dataprep import make_contour
import numpy as np
import glob

def contourGaussian(file,channel):
    # Make contour and gaussian
    f = np.genfromtxt(file,delimiter=',')
    x = f[:,1]
    y = f[:,2]
    data = f[:,channel+2]
    make_contour(x,y,data)
    params = jGaussian(x,y,data)
    return params

#%% Start out
dir_in = '../Data/july19/static/analysis/'
filelist = glob.glob(dir_in + '*.csv')
file = filelist[2]
params = []
#c1_param = []
#c2_param = []
#c3_param = []
#c4_param = []

params = contourGaussian(file,3)


#for file in filelist:
#    f = np.genfromtxt(file,delimiter=',')
#    x = f[:,1]
#    y = f[:,2]
#    z = f[:,3] # This is channel 1
#    c1 = f[:,3]
#    c2 = f[:,4]
#    c3 = f[:,5]
#    c4 = f[:,6]

#    make_contour(x,y,z) # We can easily make contour. Now I want to model a 2D Gaussian
    # Try our custom function
#    c1_param.append(jGaussian(x,y,c1))
#    c2_param.append(jGaussian(x,y,c2))
#    c3_param.append(jGaussian(x,y,c3))
#    c4_param.append(jGaussian(x,y,c4))
#    param_new = jGaussian(x,y,z)
#    params.append(param_new)
    
#%% Look at the results
#c1 = np.asarray(c1_param)
#c2 = np.asarray(c2_param)
#c3 = np.asarray(c3_param)
#c4 = np.asarray(c4_param)