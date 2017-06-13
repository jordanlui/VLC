# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:59:16 2017

@author: Jordan
Trilateration function
Accepts 3 coordinates and their radii, finds trilateration point in 2D system
For my research: I just need to confidently find the radius distance points
Input order [coord1,radius1,coord2,radius2,coord3,radius3]
Equations explained here:
https://math.stackexchange.com/questions/884807/find-x-location-using-3-known-x-y-location-using-trilateration

"""
from __future__ import division

def trilateration(c1,r1,c2,r2,c3,r3):
    x1, y1 = c1
    x2, y2 = c2
    x3, y3 = c3
    
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
    if y > max(y1,y2,y3) or y < min(y1,y2,y3) or x > max(x1,x2,x3) or x < min(x1,x2,x3):
        print 'No Real intersection'
        return 0
        
    return x,y
    
# Test the script
x,y = trilateration([5,5],7,[5,15],6,[20,10],12)
#x,y = trilateration([5,5],1,[-50,1],1,[200,10],1)

print x,y