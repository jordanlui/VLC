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
#import matplotlib.pyplot as plt
from dataprep import dataprep, unison_shuffled_copies

# Constants and parameters
path ='../Data/july6/analysis/' # Main working directory
segment = 0.2 # Amount split for training
seed = 0 # Random seed value
twidth = 2 # This is the number of label "t" columns in the front of the x-matrix

output_dir = path # Directory for results output
output_file = 'analysis.csv' # Analysis Summary file
output_path = os.path.join(output_dir,output_file)

# Functions are in dataprep.py

# Load data
x = np.genfromtxt(os.path.join(path,"x.csv"),delimiter=',') # The real x data
x = x[:,1:] # Cut out time values for now
#training_segment = int(segment*len(x))


#t1 = np.genfromtxt(os.path.join(path,"t.csv"),delimiter=',') # The real x data
[x_train, x_test, t_train, t_test] = dataprep(x,seed,segment,twidth)


# Model
regr = linear_model.LinearRegression()
regr.fit(x_train, t_train)

# Coefficients of fit
#print 'Coefficients: \n', regr.coef_

# Mean Error
print('MSE: %.2f' % np.mean((regr.predict(x_test) - t_test) **2))
print('Variance: %.2f' %regr.score(x_test,t_test))