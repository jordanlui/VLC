# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:35:50 2017

@author: Jordan

# Imports TDMS Files

See more here:
# Imports TDMS Files

"""


# Libraries
from nptdms import TdmsFile
import numpy as np
import os, glob, re, csv
# Parameters
path = "../Data/april14table/"
fileout = '../analysis/xytest_april14.csv'

# Main Code
#filename = "20_2017_03_10_16_43_46.tdms"
def tdmsfunc1(path,filename):
    # Accepts a filename and path. Opens TDMS file and extracts single column of data.
    # Import file    
    tdms_file = TdmsFile(filename)
    #specify the channel
    channel = tdms_file.object('Untitled','Dev2/ai0')
    # Does following line convert to a data object?    
    data = channel.data
#    time = channel.time_track()
    datamean = np.mean(data)
    datastd = np.std(data)
#    print 'mean of data is',average
#    print 'distance is', distance,'mm'
#    print 'filename is',filename
    return datamean, datastd

def tdmsfuncapr14(filename):
    # Accepts a filename and path. Opens TDMS file and extracts 4 columns of data, as per April 14 tests.
    # Load the file    
    tdms_file = TdmsFile(filename)
    # Specify the channel to load. Format is tab, and then channel string name
    channel1 = tdms_file.object('Untitled','40khz')
    channel2 = tdms_file.object('Untitled','100khz')
    channel3 = tdms_file.object('Untitled','10khz')
    channel4 = tdms_file.object('Untitled','1khz')
#    time= tdms_file.object('Untitled','Time*')    
    c1 = channel1.data
    c2 = channel2.data
    c3 = channel3.data
    c4 = channel4.data
    
    mean1 = np.mean(c1)
    mean2 = np.mean(c2)
    mean3 = np.mean(c3)
    mean4 = np.mean(c4)
    
#    print 'mean of data is',average
#    print 'distance is', distance,'mm'
#    print 'filename is',filename
    return mean1,mean2,mean3,mean4

def exportresults(filename,data):
#    file_exists = os.path.isfile(filename)
    with open(filename,'ab') as csvfile:
        logwriter = csv.writer(csvfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow(data)
#        filednames = ['']
#        logwriter = csv.DictWriter(csvfile,fieldnames=filednames)
#        if not file_exists:
#            logwriter.writeheader()
#        logwriter.writerow()

# Do this for all files in directory

filelist = glob.glob(os.path.join(path,'*tdms'))
# April 2017 Analysis
data = []
for file in filelist:
    
    filename = os.path.split(file)[1] #just filename, cut out path
    # Extract the XY coordinate values. Allows us to check file format also
    datalabelsearch =  re.match(r'X([0-9-]{1,3})Y([0-9-]{1,3}).tdms',filename,re.M|re.I)   
    # If file format correct, we can proceed
    if datalabelsearch:
        # Set x and y coordinate values. Note these are integers for now
        # Done on an optical breadboard with 1 inch increments
        x = datalabelsearch.group(1)
        x = int(x)
        y = datalabelsearch.group(2)
        y = int(y)
        [mean1,mean2,mean3,mean4] = tdmsfuncapr14(file)
#        print filename, x,y, mean1,mean2,mean3,mean4
        data.append([filename, x,y, mean1,mean2,mean3,mean4])
#        print dataoutput
        
    else:
        print 'not a real file, skip over'
    
dataoutput = np.asarray(data)
# Let's save our values to a file
#exportresults(fileout,data)
with open(fileout,'wb') as f:
    np.savetxt(f,data,fmt='%s',delimiter=',')

# March file analysis 2017

#for file in filelist:
#    filename = os.path.split(file)[1] #just filename, cut out path
#    # Extract the distance value, which is the positive or negative value at start of our string. Should alway be followed by an underscore
#    distancematch = re.match(r'([0-9-]{1,})_.*tdms',filename,re.M|re.I)
#    if distancematch:
#        distance = distancematch.group(1)        
#        [average,std] = tdmsfunc1(path,file)
#    else:
#        print 'not a real file, skip over'
#    print filename, distance, average, std
#    dataoutput = np.asarray([distance,average,std])
#    # Let's save our values to a file
#    exportresults(fileout,dataoutput)
    