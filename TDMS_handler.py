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
path = "../Data/april19table/"
pathout = '../analysis/april19/'
fileout = 'xytest_averages.csv'
fileout = os.path.join(pathout,fileout)


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
    channel1 = tdms_file.object('Untitled','1khz')
    channel2 = tdms_file.object('Untitled','10khz')
    channel3 = tdms_file.object('Untitled','40khz')
    channel4 = tdms_file.object('Untitled','100khz')
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

def exportresults(filename,data):
#    file_exists = os.path.isfile(filename)
    with open(filename,'wb') as csvfile:
        logwriter = csv.writer(csvfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow(data)
#        filednames = ['']
#        logwriter = csv.DictWriter(csvfile,fieldnames=filednames)
#        if not file_exists:
#            logwriter.writeheader()
#        logwriter.writerow()
        

# Do this for all files in directory

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
        x = datalabelsearch.group(1)
        x = int(x)
        y = datalabelsearch.group(2)
        y = int(y)
        # Use function to return mean values and the entire data sets
        [mean1,mean2,mean3,mean4,c1,c2,c3,c4] = tdmsfuncapr14(file)
        # Format classname in a format we desire        
        classname = 'x'+str(x)+'y'+str(y)
        # Store this to a label file
        labels.append(classname)
        # Generate column label for data
        labelcolumn = label * np.ones((len(c1),1))
        # Write mean values to data array
        data.append([filename, x,y, mean1,mean2,mean3,mean4])
        # Concatenate our data
        newdata = np.concatenate((labelcolumn,c1,c2,c3,c4),axis=1)
        # Temporarily cut our new data for faster analysis and loading
        newdata = newdata[0:300,:]
        
        data2.append(newdata)
        
        label = label+1
#        exportresults(os.path.join(pathout,"x.csv"),newdata)
#        print dataoutput
        
    else:
        print 'not a real file, skip over'
    
dataoutput = np.asarray(data)
data2out = np.vstack(data2)
with open(os.path.join(pathout,"x.csv"),'wb') as f:
    np.savetxt(f,data2out,fmt='%s',delimiter=',')

# Let's save our values to a file
#exportresults(fileout,dataoutput)
#exportresults(os.path.join(pathout,"labels.csv") , labels)
with open(fileout,'wb') as f:
    np.savetxt(f,dataoutput,fmt='%s',delimiter=',')
with open(os.path.join(pathout,"labels.csv"),'wb') as f:
    np.savetxt(f,labels,fmt='%s',delimiter=',')

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
    