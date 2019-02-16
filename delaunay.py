#!/usr/bin/python

import cv2
import numpy as np
import random

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
def draw_delaunay(img, subdiv, delaunay_color):

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1,0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, 0)
     
 
if __name__ == '__main__':

    filename = 'olivia2.jpg'

    # Read in image
    img = cv2.imread(filename);

    # Rectangle for Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);

    # Create array of points
    points = [];

    # Read in points from text file
    with open("olivia2.jpg.txt") as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Draw triangles
    draw_delaunay(img, subdiv, (255, 255, 255));

    cv2.imshow("Delaunay Triangulation", img)
    cv2.waitKey(0)
