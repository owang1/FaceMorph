#! /usr/bin/env python

import sys
import numpy as np
import cv2
import math
#import scipy
#from scipy.stats import norm
#from scipy.signal import convolve2d

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    

    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri
        

def adjust_gamma(image, gamma = 1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def split_rgb(image):
  red = None
  green = None
  blue = None
  (blue, green, red) = cv2.split(image)
  return red, green, blue

def generating_kernel(a):
  w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
  return np.outer(w_1d, w_1d)

def gauss_pyramid(image, levels):
  output = []
  output.append(image)
  tmp = image
  for i in range(0, levels):
    tmp = image
    for i in range(0, levels):
      tmp = ireduce(tmp)
      output.append(tmp)
    return output

def ireduce(image):
  out = None
  kernel = generating_kernel(0.4)
  outimage = scipy.signal.convolve2d(image,kernel,'same')
  out = outimage[::2, ::2]
  return out

def iexpand(image):
  out = None
  kernel = generating_kernel(0.4)
  outimage = np.zeros(image.shape[0]*2, image.shape[1]*2)
  outimage[::2,::2]=image[:,:]
  out = 4*scipy.signal.convolve2d(outimage,kernel,'same')
  return out

def lap1_pyramid(gauss_pyr):
  output = []
  k = len(gauss_pyr)
  for i in range(0,k-1):
    gu = gauss_pyr[i]
    egu = iexpand(gauss_pyr[i+1])
    if egu.shape[0] > gu.shape[0]:
      egu = np.delete(egu, (-1), axis=0)
    if egu.shape[1] > gu.shape[1]:
      egu = np.delete(egu, (-1), axis=1)
    output.append(gu - egu)
  output.append(gauss_pyr.pop())
  return output

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    

if __name__ == '__main__' :
    
    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3 :
        print >>sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
        sys.exit(1)

    # Read images
    filename1 = '3_morph.jpg'
    filename2 = '3a.jpg'
    
    img1 = cv2.imread('./morphs/' + filename1);
    img2 = cv2.imread('./pairs/' + filename2);

    img1Warped = np.copy(img2);    
    
    # Read array of corresponding points
    points1 = readPoints('./output/' + filename1 + '.txt')
    points2 = readPoints('./output/' + filename2 + '.txt') 

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)
          
    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])
   
    # Find delanauy traingulation for convex hull points
    sizeImg2 = img2.shape    
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
     
    dt = calculateDelaunayTriangles(rect, hull2)
 
    if len(dt) == 0:
        quit()
    
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])
        
        warpTriangle(img1, img1Warped, t1, t2)
            
    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))
    
    mask = np.zeros(img2.shape, dtype = img2.dtype)  
    
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    
    r = cv2.boundingRect(np.float32([hull2]))    
    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
    '''
    r1 = None
    g1 = None
    b1 = None
    r2 = None
    g2 = None
    b2 = None
    rm = None
    gm = None
    bm = None

    (r1, g1, b1) = split_rgb(img1Warped)
    (r2, g2, b2) = split_rgb(img2)
    (rm, gm, bm) = split_rgb(mask) 

    r1 = r1.astype(float)
    g1 = g1.astype(float)
    b1 = b1.astype(float)

    r2 = r2.astype(float)
    g2 = g2.astype(float)
    b2 = b2.astype(float)

    rm = rm.astype(float)/255
    gm = gm.astype(float)/255
    bm = bm.astype(float)/255

    min_size = min(r1.shape)
    depth = int(math.floor(math.log(min_size, 2)))

    gauss_pyr_maskr = gauss_pyramid(rm, depth)
    gauss_pyr_masg = gauss_pyramid(gm, depth)
    gauss_pyr_maskb = gauss_pyramid(bm, depth)

    gauss_pyr_image1r = gauss_pyramid(r1, depth)
    gauss_pyr_image1g = gauss_pyramid(g1, depth)
    gauss_pyr_image1b = gauss_pyramid(b1, depth)
 
    gauss_pyr_image2r = gauss_pyramid(r2, depth)
    gauss_pyr_image2g = gauss_pyramid(g2, depth)
    gauss_pyr_image2b = gauss_pyramid(b2, depth)

    lap1_pyr_image1r = lap1_pyramid(gauss_pyr_image1r)
    lap1_pyr_image1g = lap1_pyramid(gauss_pyr_image1g)
    lap1_pyr_image1b = lap1_pyramid(gauss_pyr_image1b)

    lap1_pyr_image2r = lap1_pyramid(gauss_pyr_image2r)
    lap1_pyr_image2g = lap1_pyramid(gauss_pyr_image2g)
    lap1_pyr_image2b = lap1_pyramid(gauss_pyr_image2b)
 
    outpyrr = blend(lap1_pyr_image2r, lap1_pyr_image1r, gauss_pyr_maskr)
    ''' 

    '''
    alpha = float(1.2)
    beta = int(10) 

    new_image = np.zeros(img1Warped.shape, img1Warped.dtype)
  
    for y in range(img1Warped.shape[0]):
      for x in range(img1Warped.shape[1]):
        for c in range(img1Warped.shape[2]):
          new_image[y,x,c] = np.clip(alpha*img1Warped[y,x,c] + beta, 0, 255) 
    '''
    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
 
    # gamma = 1.5
    # swap = adjust_gamma(output, gamma = gamma)
    # swap2 = cv2.illuminationChange(np.uint8(img1Warped), mask, img2, .5, .5)    

    cv2.imshow("Face Swapped", output)
    cv2.imwrite('the_morph.jpg', output)
    cv2.waitKey(0) 
 
    cv2.destroyAllWindows() 
