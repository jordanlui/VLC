# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:46:21 2017

@author: Jordan

Trying linear regression models out

Need to improve abstraction in this code like my other code

"""

from __future__ import division
print 'Start of script'
# Libraries
import numpy as np
import os, random
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from algorithm import dataprep

# Constants and parameters
path ='../Data/july6/analysis/' # Main working directory
segment = 0.3 # Amount split for training

output_dir = path # Directory for results output
output_file = 'analysis.csv' # Analysis Summary file
output_path = os.path.join(output_dir,output_file)

# Functions
def unison_shuffled_copies(a, b):
    # Shuffles to arrays together in unison
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Load data
x1 = np.genfromtxt(os.path.join(path,"x.csv"),delimiter=',') # The real x data
t1 = np.genfromtxt(os.path.join(path,"t.csv"),delimiter=',') # The real x data

#a = t
#b = x
#c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
#np.random.shuffle(c)
#a2 = c[:, :a.size//len(a)].reshape(a.shape)
#b2 = c[:, a.size//len(a):].reshape(b.shape)
training_segment = int(segment*len(x1))

# Shuffle
x,t = unison_shuffled_copies(x1,t1)
#x = x1
#t = t1
# Split data
X_test = x[:training_segment,:]
X_train = x[training_segment:,:]
y_test = t[:training_segment,:]
y_train = t[training_segment:,:]

# Model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Coefficients of fit
print 'Coefficients: \n', regr.coef_

# Mean Error
print('MSE: %.2f' % np.mean((regr.predict(X_test) - y_test) **2))
print('Variance: %.2f' %regr.score(X_test,y_test))

# Plot outputs
#plt.scatter(X_test, y_test, color='black')
#plt.plot(X_test, regr.predict(X_test), color='blue',linewidth=3)
#plt.xticks(())
#plt.yticks(())
#plt.show()