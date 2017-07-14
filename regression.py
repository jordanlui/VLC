# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:46:21 2017

@author: Jordan

July 2017: Trying linear regression models out

Need to improve abstraction in this code like my other code

Issue: Using dataprep function from algorithm.py but it's running whole script?

"""

from __future__ import division
print 'Start of script'
# Libraries
import numpy as np
import os
#from sklearn import datasets
#from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from dataprep import dataprep, unison_shuffled_copies

# Constants and parameters
path ='../Data/july14/corners2/analysis/' # Main working directory
segment = 0.9 # Amount split for training
seed = 0 # Random seed value
twidth = 2 # This is the number of label "t" columns in the front of the x-matrix
scale_table = 1.333 # This is the scaling factor of table at 805mm sys height. Units are pixels/mm

output_dir = path # Directory for results output
output_file = 'analysis.csv' # Analysis Summary file
output_path = os.path.join(output_dir,output_file)

def model(x_train,x_test,t_train,t_test):
    #%% Model
    # Round t values (coordinates) since the coord sampling is troublesome
    t_train = np.round(t_train,decimals=0)
    t_test = np.round(t_test,decimals=0)
    
    regr = linear_model.LinearRegression()
    regr.fit(x_train, t_train)
    
    #%% Results
    # Coefficients of fit
    #print 'Coefficients: \n', regr.coef_
    coeff = regr.coef_
    # Mean Error
    print('MSE: %.2f' % np.mean((regr.predict(x_test) - t_test) **2))
    print('Variance: %.2f' %regr.score(x_test,t_test))
    
    # Try Euclidean distance error on the system predictions
    x_pred = regr.predict(x_test)
    diff = (x_pred - t_test)**2 # Square errors
    diff = np.sqrt(np.sum(diff,axis=1)) # Sum the square error and sqrt. Euclidean distance error
    error_mean = np.mean(diff) # Per pixel error mean
    error_mean = error_mean/scale_table # Error in mm
    print 'mean error is',error_mean,'mm'
    diff_mm = diff/scale_table # This is the mm error values
    error_min = np.min(diff_mm)
    error_max = np.max(diff_mm)
    error_med = np.median(diff_mm)
    print 'min error (mm) is',error_min,'max error',error_max,'median',error_med
    
    # Histogram of analysis
    #np.histogram(diff_mm,bins=10)
    plt.figure()
    plt.hist(diff_mm,bins='auto')
    title = 'Histogram of error (mm)'
    plt.title(title)
    plt.ylabel('Occurrences')
    plt.xlabel('Error value (mm)')
    plt.show()
    
    return error_mean, error_min, error_max, error_med
    

# Functions are in dataprep.py

#%% Main Script
# Load data
# Get all the files in the folder
filelist = glob.glob(path + '*.csv')
numfiles = len(filelist)
seed = numfiles

e_mean = []
e_min = []
e_max = []
e_med = []
for i in range(0,numfiles): # Iterate through and generate train and test data
    x_train = []
    singlepath = filelist[i] # Single path
    print os.path.basename(singlepath), ' is the file we test on'
    otherpaths = filelist[:i] + filelist[i+1:] # Rest of paths
    
    x_test = np.genfromtxt(singlepath,delimiter=',') # The real x data
    t_test = x_test[:,1:3]
    x_test = x_test[:,3:]
    
    for apath in otherpaths:
        xtemp = np.genfromtxt(apath,delimiter=',')
        x_train.append(xtemp)
        
    x_train = np.vstack(x_train)
    t_train = x_train[:,1:3]
    x_train = x_train[:,3:]
    
    # Run model
    error_mean, error_min, error_max, error_med = model(x_train,x_test,t_train,t_test)
    e_mean.append(error_mean)
    e_min.append(error_min)
    e_max.append(error_max)
    e_med.append(error_med)
    
# Summarize results
print e_mean
#%% Old stuff
# Now stack files properly

#[x_train, x_test, t_train, t_test] = dataprep(x,seed,segment,twidth)
#
## Load second data to compare
#x2 = np.genfromtxt(filelist[0],delimiter=',') # The real x data
#x2 = x2[:,1:] # Cut out time values for now
#
#[x_train2, x_test2, t_train2, t_test2] = dataprep(x2,seed,segment,twidth)
#
##  Change up so we train one, test other
#x_test = x_test2
#t_test = t_test2



##%% Model
## Round t values (coordinates) since the coord sampling is troublesome
#t_train = np.round(t_train,decimals=0)
#t_test = np.round(t_test,decimals=0)
#
#regr = linear_model.LinearRegression()
#regr.fit(x_train, t_train)
#
##%% Results
## Coefficients of fit
##print 'Coefficients: \n', regr.coef_
#coeff = regr.coef_
## Mean Error
#print('MSE: %.2f' % np.mean((regr.predict(x_test) - t_test) **2))
#print('Variance: %.2f' %regr.score(x_test,t_test))
#
## Try Euclidean distance error on the system predictions
#x_pred = regr.predict(x_test)
#diff = (x_pred - t_test)**2 # Square errors
#diff = np.sqrt(np.sum(diff,axis=1)) # Sum the square error and sqrt. Euclidean distance error
#error_mean = np.mean(diff) # Per pixel error mean
#print 'mean error is',error_mean/scale_table,'mm'
#diff_mm = diff/scale_table # This is the mm value error
#print 'min error (mm) is',np.min(diff_mm),'max error',np.max(diff_mm),'median',np.median(diff_mm)
#
## Histogram of analysis
##np.histogram(diff_mm,bins=10)
#plt.figure()
#plt.hist(diff_mm,bins='auto')
#title = 'Histogram of error (mm)'
#plt.title(title)
#plt.ylabel('Occurrences')
#plt.xlabel('Error value (mm)')
#plt.show()
