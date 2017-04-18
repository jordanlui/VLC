# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:10:27 2017

@author: Jordan
"""
import re

file1 = "20_2017_03_10_16_43_46.tdms"
file2 = "-10_2017_03_10_16_43_46.tdms"
file3 =  "1DTranslation_20_2017_03_10_16_43_46.tdms"
file4 = "../Data/proto3_mar10\-10_2017_03_10_16_39_47.tdms"

#p = re.compile("lalala(I want this part)")
#print p.match("lalalaI want this partlalala and more stuff20_2017_03_10_16_43_46.tdms").group(1)

#p2 = re.compile("lala*")
#print p2.match("lalalaI want this partlalala and more stuff20_2017_03_10_16_43_46.tdms").group(1)
#
#p3 = re.compile("20*")

# Trying with re.match
match1 = re.match(r'(20).*tdms',file1,re.M|re.I)
if match1:
    print 'searching for literal 20:',match1.group(1)
else:
    print 'no match'

match2 = re.match(r'([0-9-]{2,})_.*tdms',file4,re.M|re.I)
# match22 = re.findall('([0-9-]{2,})',file2) # Other easier option which works
#print match22[0]
if match2:
    print match2.group(1)
else:
    print 'no match'
    
#match3 = re.match(r'',file3,re.M|re.I)
#if match3:
#    print match3.group(1)
#else:
#    print 'no match'