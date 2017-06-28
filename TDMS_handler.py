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
# Input file directory
filename = 'circle_fast_2.tdms'
dirout = "../Data/june27/"
path = dirout + filename
# Output file parameters
pathout = '../analysis/june27/'
fileout = filename[:-5] + '.csv' # output file name of the averages file, named based on input file. Clipped to remove TDMS filename
filepathout = os.path.join(pathout,fileout)


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
    # Fix jagged edge issue, where size of x,y column often shorter than the photodiode data
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

# Main code
print 'Testing new code June 2017'
f = tdmsfuncjun(path)
print 'function finished'

# Save results to file
with open(os.path.join(pathout,fileout),'wb') as file:
        np.savetxt(file,f,fmt='%s',delimiter=',')
    