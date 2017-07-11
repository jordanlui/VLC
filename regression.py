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
path ='../Data/july6/analysis/' # Main working directory
segment = 0.9 # Amount split for training
seed = 0 # Random seed value
twidth = 2 # This is the number of label "t" columns in the front of the x-matrix
scale_table = 1.333 # This is the scaling factor of table at 805mm sys height. Units are pixels/mm

output_dir = path # Directory for results output
output_file = 'analysis.csv' # Analysis Summary file
output_path = os.path.join(output_dir,output_file)

# Functions are in dataprep.py

#%% Main Script
# Load data
x = np.genfromtxt(os.path.join(path,"x.csv"),delimiter=',') # The real x data
x = x[:,1:] # Cut out time values for now

[x_train, x_test, t_train, t_test] = dataprep(x,seed,segment,twidth)

# Round t values since the time sampling is troublesome
t_train = np.round(t_train,decimals=0)
t_test = np.round(t_test,decimals=0)


#%% Model
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
print 'mean error is',error_mean/scale_table,'mm'
diff_mm = diff/scale_table # This is the mm value error
print 'min error (mm) is',np.min(diff_mm),'max error',np.max(diff_mm),'median',np.median(diff_mm)

# Histogram of analysis
#np.histogram(diff_mm,bins=10)
plt.hist(diff_mm,bins='auto')
plt.title('Histogram of error (mm)')
plt.ylabel('Occurrences')
plt.xlabel('Error value (mm)')
plt.show()

##%% Individual predictions
##xx,tt = regr.predict(x_train[0:2,:])
##print np.dot(coeff,np.transpose(x_train[0,:]))
#error = []
#for i in range(0,len(x_test)):
#    xx,yy = t_test[i,:] # Real values
#    coord = x_test[i,:] # Coordinates
#    coord = np.reshape(coord,(4,1)).transpose()
#    px,py = np.dot(coeff,np.transpose(coord)) # Prediction. But this seems wrong.
#    px,py = regr.predict(coord)
#    diff = np.sqrt((xx - px)**2 + (yy-py)**2) # Euclidean distance between points
#    error.append(diff)
#
## Average of result
#print 'mean value is',np.mean(error)
#print 'max is',np.max(error)
#print 'min is',np.min(error)
