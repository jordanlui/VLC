# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 10:22:18 2017

@author: Jordan
"""

print 'ciao a tutti'

# Libraries
import numpy as np

# Constants and parameters
path ='../Analysis/may29/' # Main working directory
seed = 1
segment = 0.7 # Segment of data we train on 
twidth = 2 # We have two columns of label data in front of x matrix from May 29


output_dir = '../Analysis/may29/' # Directory for results output
output_file = 'xytable_analysis.csv' # Analysis Summary file
output_path = os.path.join(output_dir,output_file)

# Load data
x = np.genfromtxt(os.path.join(path,"x.csv"),delimiter=',')


# Functions
def dataprep(x,seed,segment,twidth):
    random.seed(a=seed)
    x = np.asarray(random.sample(x,len(x)))
    
    # Find index at which we segment our data    
    segindex = int(len(x)*segment)
    
    # Segment our data (May29 data set formatting rules)
    
    # Grab X values. 
    #twidth = 2 # We have two columns of label data in front of x matrix from May 29
    x_train = x[:segindex,twidth:]
    x_test = x[segindex:,twidth:]
    
    # Segment T values 
    t_train = x[:segindex,:twidth]
    t_test = x[segindex:,:twidth]
    
    # Reshape t to proper array if twidth was 1, so we are not dealing with list
    if twidth == 1:
        t_train = np.reshape(t_train,(t_train.shape[0],twidth))
        t_test = np.reshape(t_test,(t_test.shape[0],twidth))

    return x_train,x_test,t_train,t_test
def model():
    # Accepts a test value. Model will search for nearest coordinate in xmatrix list and return the coordinates
    return 0
def score():
    # Accepts prediction and real value. Returns a 
    return 0

# Main loop
# Import our data to get started
[x_train, x_test, t_train, t_test] = dataprep(x,seed,segment,twidth)


# Loop though a test set, evaluate accuracy for each one

# Data analysis
# Some method to average or quantify error through the whole test set
