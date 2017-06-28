# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:35:50 2017

@author: Jordan

# Imports TDMS Files

See more here:
# Imports TDMS Files and makes CSV File of average values
Currently the script outputs raw values, which were low pass filtered by the LabVIEW script
Next step: Normalize values before create the averages file, and likely enable high accuracies

2017 June
Updated code for the continuous motion tracking method


"""


# Libraries
from nptdms import TdmsFile
import numpy as np
import os, glob, re, csv
# Parameters
path = "../Data/june27/circle_slow_1.tdms"
pathout = '../analysis/jun27/'
fileout = 'averages.csv' # output file name of the averages file
fileout = os.path.join(pathout,fileout)


def tdmsfuncapr14(filename):
    # Accepts a filename and path. Opens TDMS file and extracts 4 columns of data, as per April 14 tests.
    # Load the file    
    tdms_file = TdmsFile(filename)
    # Specify the channel to load. Format is tab, and then channel string name
    channel1 = tdms_file.object('Untitled','1khz (Filtered)')
    channel2 = tdms_file.object('Untitled','10khz (Filtered)')
    channel3 = tdms_file.object('Untitled','40khz (Filtered)')
    channel4 = tdms_file.object('Untitled','100khz (Filtered)')
#    time= tdms_file.object('Untitled','Time*')    
    c1 = channel1.data
    c2 = channel2.data
    c3 = channel3.data
    c4 = channel4.data
    c1 = np.reshape(c1,(len(c1),1))
    c2 = np.reshape(c2,(len(c2),1))
    c3 = np.reshape(c3,(len(c3),1))
    c4 = np.reshape(c4,(len(c4),1))
    
    mean1 = np.mean(c1)
    mean2 = np.mean(c2)
    mean3 = np.mean(c3)
    mean4 = np.mean(c4)
    
#    print 'mean of data is',average
#    print 'distance is', distance,'mm'
#    print 'filename is',filename
    return mean1,mean2,mean3,mean4,c1,c2,c3,c4
def tdmsfuncjun(filename):
    # Accepts files from June 2017, which now have motion tracker data incorporated
    # Goal: Create a matrix containing labels and x values [time,x,y,a,b,c,d]
    
    # Load the file
    tdms_file = TdmsFile(filename) # Load the file
    
    # Load the individual data channels
    
    c1 = tdms_file.object('Untitled','1khz (Filtered)').data
    c2 = tdms_file.object('Untitled','10khz (Filtered)').data
    c3 = tdms_file.object('Untitled','40khz (Filtered)').data
    c4 = tdms_file.object('Untitled','100khz (Filtered)').data
    
    # Load time data and coord data. More painful
    x = tdms_file.object('Untitled','Resampled').data
    y = tdms_file.object('Untitled','Resampled 1').data
    t = tdms_file.object('Untitled','1khz (Filtered)').time_track()
    
    # Reshape data
    c1 = np.reshape(c1, (len(c1),1))
    c2 = np.reshape(c2, (len(c2),1))
    c3 = np.reshape(c3, (len(c3),1))
    c4 = np.reshape(c4, (len(c4),1))
    x = np.reshape(x, (len(x),1))
    y = np.reshape(y, (len(y),1))
    t = np.reshape(t, (len(t),1))
    
    
    print 'length compare', len(x), len(c1)
    # Fix jagged edge issue
    if len(x) < len(c1):
        # The fix jagged edge issue
        jagged = len(x)
        c1 = c1[:jagged]
        c2 = c2[:jagged]
        c3 = c3[:jagged]
        c4 = c4[:jagged]
        t = t[:jagged]
        
    # Format as a new array
    f = np.concatenate((t,x,y,c1,c2,c3,c4), axis=1)
    
    # Return result
    return f   
    
    
    
    


def discretefiles():
    # Processes files in a directory, from our discrete recording method in April 2017
    filelist = glob.glob(os.path.join(path,'*tdms'))
    
    data = [] # This is array where we store the average values for each position
    data2 = [] # Array where we store all data from the file and combine
    labels = [] # we store the order of labels here
    label = int(1)
    for file in filelist:
        
        filename = os.path.split(file)[1] #just filename, cut out path
        # Extract the XY coordinate values. Allows us to check file format also
        datalabelsearch =  re.match(r'X([0-9-]{1,3})Y([0-9-]{1,3}).tdms',filename,re.M|re.I)   
        # If file format correct, we can proceed
        if datalabelsearch:
            # Set x and y coordinate values. Note these are integers for now
            # Done on an optical breadboard with 1 inch increments
            # Extract x and y coord
            x = datalabelsearch.group(1)
            x = int(x)
            y = datalabelsearch.group(2)
            y = int(y)
            
            # Use function to return mean values and the entire data sets in extracted columns
            [mean1,mean2,mean3,mean4,c1,c2,c3,c4] = tdmsfuncapr14(file)
            
            # Format classname in a format we desire        
            classname = 'x'+str(x)+'y'+str(y)
            # Try making a new row that we can work with
            labelsrow = [classname,x,y] # This is the string coordinate and numeric x,y coordinates
            
            # Append the coordinates into a table
            labels.append(labelsrow) # This is the name table of all possible labels and coordinates
            
            # Generate column label for data so we can generate matrix
            label = np.reshape(labelsrow[1:],([1,2])) # The actual x,y coordinate values in array
            labelcolumn = label * np.ones((len(c1),1)) # Make into a matching array size to mate with columns
    
            # Write mean values to data array
            data.append([filename, x,y, mean1,mean2,mean3,mean4])
            
            # Concatenate our data into a matrix
            newdata = np.concatenate((labelcolumn,c1,c2,c3,c4),axis=1)
     
            # If desired, we can cut out some data in each loop to decrease x-matrix size
    #        newdata = newdata[100:700,:]
            
            # Append values into data2, which contains all data points
            data2.append(newdata)
            
    #        label = label+1 # increment our archaic label ordinal system. soon to be removed.
    #        exportresults(os.path.join(pathout,"x.csv"),newdata)
    #        print dataoutput
            
        else:
            print 'not a real file, skip over'
    
    #  average power values     
    power_average = np.asarray(data) # This is the table of average power values based on position
    power_all = np.vstack(data2) # This is the full table of power values 
    
    # Save x matrix of all values to file
    with open(os.path.join(pathout,"x.csv"),'wb') as f:
        np.savetxt(f,power_all,fmt='%s',delimiter=',')
    
    # Let's save our values to a file - not currently working - so abandoned.
    #exportresults(fileout,dataoutput)
    #exportresults(os.path.join(pathout,"labels.csv") , labels)
    
    # Save data averages to a file
    with open(fileout,'wb') as f:
        np.savetxt(f,power_average,fmt='%s',delimiter=',')
    
    # Save our labels to a file
    with open(os.path.join(pathout,"labels.csv"),'wb') as f:
        np.savetxt(f,labels,fmt='%s',delimiter=',')

# Main code
print 'Testing new code June 2017'
testarray = tdmsfuncjun(path)
print 'function finished'
#print testarray.shape
    