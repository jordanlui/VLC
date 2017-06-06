# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 10:22:18 2017

@author: Jordan

Localization Algorithm

2017 June 6: Algorithm doing basic match against table of average values for the 4 channel power values.
Data source: May 29 data recording

"""

print 'Start of script'

# Libraries
import numpy as np
import os, random

# Constants and parameters
path ='../Analysis/may29/' # Main working directory
seed = 1
segment = 0.7 # Segment of data we train on 
twidth = 2 # We have two columns of label data in front of x matrix from May 29
tableconversion = 1 * 25.4 # unit length on optical table is 2.54 cm or 25.4 mm


output_dir = '../Analysis/may29/' # Directory for results output
output_file = 'xytable_analysis.csv' # Analysis Summary file
output_path = os.path.join(output_dir,output_file)

# Load data
x = np.genfromtxt(os.path.join(path,"x.csv"),delimiter=',') # The real x data
database = np.genfromtxt(os.path.join(path,"xavg.csv"),delimiter=',') # The database of signal strengths


# Functions
def dataprep(x,seed,segment,twidth):
    # Takes our x matrix and shuffles, segments, and extracts label values
    # Shuffle data
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
    
def min_in_list(column):
    # Finds index of minimum value in a list (no interpolation)
    # Initialize empty search variable for the 
    minsofar = column[0]
    mindex = [0]
    # Loop through, but skip last iteration because we'll go too far
    for i in range(0,int(column.shape[0])-1):
        
        if column[i+1] < minsofar:
            # Smaller value found. Update minsofar and mindex
            minsofar = column[i+1]
            mindex = i+1 # This is the index of smallest value found so far
    return mindex
    
def model_database(database,query):
    # Accepts a test value. Model will search for nearest coordinate in database list and return the coordinates
    # Database format should be [x,y,A,B,C,D]
    # Inputs are the database, the column to search in, and the queried value (power value)
    
    # Determine number of channels
    channels = int(query.shape[0])
    
    # Allocate lists to hold our coordinates
    xs=[] 
    ys=[]
    
    # loop through each channel (0,1,2,3)
    for channel in range(0,channels):
        
        column = database[:,channel+2] # Grab the single column
        query_single = query[channel] # Grab single query value
        column_diff = abs(column - query_single)
        
        # Use index algorithm to find the row with power value closest to query value
        index = min_in_list(column_diff)
        # Use index value to find corresponding x and y values. Append to list
        xfound = database[index,0]
        yfound = database[index,1]
        xs.append(xfound)
        ys.append(yfound)
#        print channel, query_single, index
    return xs,ys
def score(tx,ty,px,py):
    # Accepts prediction and real value. Returns a SSE error value
    # Inputs are [truex,truey,predx,predy]
    # Note that we may receive vector of predictions
    import numpy as np
    if len(px) > 1 or len(py) > 1:
        x = np.mean(px)
        y = np.mean(py)
    else:
        x = px
        y = py
    sse = float((tx-x)**2 + (ty-y)**2)
    return sse

# Main loop
# Import our data to get started. 
[x_train, x_test, t_train, t_test] = dataprep(x,seed,segment,twidth)


# Loop though a test set, evaluate accuracy for each one
# Define empty error object
error = []
for i in range(0,200):

    sampleindex = i
    
    # Extract our query value
    query = x_test[sampleindex,:]
    xreal = t_test[sampleindex,0]
    yreal = t_test[sampleindex,1]
    
    [xs, ys] = model_database(database,query)
    
    xavg = np.mean(xs)
    yavg = np.mean(ys)
    
#    print 'predict coord', xavg,yavg
#    print 'real is',xreal,yreal
    
    # Score our result. 
    # Force a float because of some outputs that occur
    error_single = score(xreal,yreal,xs,ys)
#    print 'error is',score # This actual error score would be in inches
#    print 'spatial error is',score*tableconversion, 'mm'
    error.append(error_single)

# Data analysis
# Some method to average or quantify error through the whole test set
print 'mean error is',np.mean(error)*tableconversion,'mm'

print 'end of analysis'
