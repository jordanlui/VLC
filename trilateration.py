# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:59:16 2017

@author: Jordan
Trilateration function
Accepts 3 coordinates and their radii, finds trilateration point
For my research: I just need to confidently find the radius distance points
Input order [coord1,radius1,coord2,radius2,coord3,radius3]
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
    return x,y

# Test the script
x,y = trilateration([5,5],7,[5,15],6,[20,10],12)
print x,y