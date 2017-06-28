# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 10:22:18 2017

@author: Jordan

Localization Algorithm

2017 June 6: Algorithm doing basic match against table of average values for the 4 channel power values.
Data source: May 29 data recording
Very rudimientary - and probably has error resulting from the non function nature of Gaussian distribution.

June 14: Assume a generally circular radiation symmetry. Do basic finger printing type analysis.
From each channel, find coordinate of closest power value. Find a distance value d.
Do a triangulation routine. Get 4 possible points and average the result.

"""
from __future__ import division
print 'Start of script'

# Libraries
import numpy as np
import os, random

#from trilateration import trilateration # This is a tr

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
#x = np.genfromtxt('../Analysis/may29/x.csv',delimiter=',')
database = np.genfromtxt(os.path.join(path,"xavg.csv"),delimiter=',') # The database of signal strengths

# Normalize our data - 
#x[:,2:] = (x[:,2:] - x[:,2:].min(0)) / x[:,2:].ptp(0)
#database[:,2:] = (database[:,2:] - database[:,2:].min(0)) / database[:,2:].ptp(0)

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
def max_in_list(column):
    # Finds max in list
    maxsofar = column[0]
    maxdex = 0
    # Loop through our list and search
    for i in range(0,int(column.shape[0])-1):
        if column[i+1] > maxsofar:
            # Bigger value found. Update our record
            maxsofar = column[i+1]
            maxdex = i + 1
    return maxdex
    
def trilateration(c,r):
    # Trilateration based on 3 coordinates in c and 3 radii in r
    # C input should be a list
#    if len(c) != 3 or len(r) != 3:
#        print 'Improper input.. 3 element list of coordinates expected for c'
#        break
    x1, y1 = c[0]
    x2, y2 = c[1]
    x3, y3 = c[2]
    r1 = r[0]
    r2 = r[1]
    r3 = r[2]
    c = 0.1 # Tolerance value for our checksum
    
    # Trilateration Linear equation parameters
    A = -2 * x1 + 2 * x2
    B = -2 * y1 + 2 * y2
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = -2 * x2 + 2 * x3
    E = -2 * y2 + 2 * y3
    F = r2**2 - r3**2 - x2**2 + x3**2 - y2**2 + y3**2

    # Obtain the actual x,y values
    x = (C * E - F * B) / (E * A - B * D)
    y = (C * D - A * F) / (B * D - A * E)
    
    # Checksum on results
#    if y > max(y1,y2,y3)  or y < min(y1,y2,y3) or x > max(x1,x2,x3) or x < min(x1,x2,x3):
#        print 'No Real intersection'
#        return 0
        
    return x,y
    
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
    
def fingerprint_trilat(database,query):
    # Accepts a signal database and query
    # Find the circle centre and radius for each transmitter, and then trilaterate our point
    # 2017-06-14. Built for 4 channel system (some hard cording probably present)
    
    # Number of channels
    channels = int(query.shape[0]) # Should be 4 channels (0,1,2,3)
    
    # Find the circle centre points
    xc = []
    yc = []
    for channel in range(0,channels):
        
        # Look for the row with the highest power value. This represents the centre
        maxdex = int(max_in_list(database[:,channel+2]))
#        print 'channel',channel,'row number',maxdex

        xc.append( float(database[maxdex,0]))
        yc.append( float(database[maxdex,1]))
    
    # Find the closest power value (basic interpolation)
    # Then calculate distance from this location to transmitter
    d = []
    for channel in range(0,channels):
        
        column = database[:,channel+2] # Grab the single column
        query_single = query[channel] # Grab single query value
        column_diff = abs(column - query_single)
        
        # Use index algorithm to find the row with power value closest to query value
        index = min_in_list(column_diff)
        # Use index value to find corresponding x and y values. Append to list
        xfound = database[index,0]
        yfound = database[index,1]
        
        # Calculate the distance from found points to the transmitters
        # Unclear if this is a solid approach.
        dist = np.sqrt( (xc[channel] - xfound)**2 + (yc[channel] - yfound)**2)
        d.append(dist)
    
    # Trilaterate. We have 4 transmitters so we can permute and do 4 times. 
    # Lazy permute option: Loop 0-3, and just ignore that coordinate when doing trilateration
    xs = []
    ys = []
    for i in range(0,channels):
        # Pick the set we'll analyze. AKA ignore one coord, and leave 3 left
        # Delete one of our points, and do analysis on remaining 3
        trilat_input_d = np.delete(d,i)
        trilat_input_x = np.delete(xc,i)
        trilat_input_y = np.delete(yc,i)
        
        # Assign values. This is a bad cast, but good enough for now
        c = [[trilat_input_x[0],trilat_input_y[0]] , [trilat_input_x[1],trilat_input_y[1]] , [trilat_input_x[2],trilat_input_y[2]]]
        r = list(trilat_input_d)
        tri_x,tri_y = trilateration(c,r)
        xs.append(tri_x)
        ys.append(tri_y)
    
#     Then take the average of our trilateration results
    xavg = np.mean(xs)
    yavg = np.mean(ys)
    print xs,ys,np.std(xs) , np.std(ys)
    
    return xavg,yavg
    
    
#    return 0 # Remove before flight
def score(tx,ty,px,py):
    # Accepts prediction and real value. Returns a SSE error value
    # Inputs are [truex,truey,predx,predy]
    # Note that we may receive vector of predictions - and should account for this
    sse = []
    import numpy as np
#    if len(px) > 1 or len(py) > 1:
##        x = np.mean(px)
##        y = np.mean(py)
#        for i in range(0,len(px)):
#            x = px[i]
#            y = py[i]
#            # Calculate error as Euclidean distance
#            sse.append(float(np.sqrt((tx-x)**2 + (ty-y)**2)))
#    else:
    x = px
    y = py
    sse = float(np.sqrt((tx-x)**2 + (ty-y)**2))
    return sse

# Main loop
# Import our data to get started. 
[x_train, x_test, t_train, t_test] = dataprep(x,seed,segment,twidth)

# Better idea to use the x_train data as the source for our database. This will require some re-processing

# Loop though a test set, evaluate accuracy for each one
# Define empty error object
error = []
for i in range(0,20):

    sampleindex = i
    
    # Extract our query value
    query = x_test[sampleindex,:]
    xreal = t_test[sampleindex,0]
    yreal = t_test[sampleindex,1]
    
    # Find predicted value
#    [xs, ys] = model_database(database,query)
    [xs, ys] = fingerprint_trilat(database,query)

    # Score our result. 
    # Force a float because of some outputs that occur
    error_single = score(xreal,yreal,xs,ys)
#    print 'error is',score # This actual error score would be in inches
#    print 'spatial error is',score*tableconversion, 'mm'
    error.append(error_single)

# Data analysis
# Some method to average or quantify error through the whole test set
errormeans = np.mean(error,axis=0)*tableconversion
print 'Overal Error is ', errormeans,'mm'

print 'end of analysis'
