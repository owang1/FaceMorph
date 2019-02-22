#!/usr/bin/env python

import cv2
import dlib
import numpy as np
import math
import sys
import os


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# for file in os.listdir('./images'):
# iterate through numbered files
for n in range(1, 4):
  img1 = cv2.imread("./pairs/" + str(n) + "a.jpg", 0)
  img2 = cv2.imread("./pairs/" + str(n) + "b.jpg", 0)

  dets1 = detector(img1);
  dets2 = detector(img2);  

  for k, d in enumerate(dets1):
    shape1  = predictor(img1, d)
  for k, d in enumerate(dets2):
    shape2  = predictor(img2, d) 
  

  vec1 = np.empty([68, 2], dtype = int)
  vec2 = np.empty([68, 2], dtype = int) 

  for b in range(68):
    vec1[b][0] = (shape1.part(b).x)
    vec1[b][1] = (shape1.part(b).y)
    vec2[b][0] = (shape2.part(b).x)
    vec2[b][1] = (shape2.part(b).y) 

  f1 = open("./output/" + str(n) + "a.jpg.txt", 'w')

  for x in range(len(vec1)):
    print(' '.join(map(str, vec1[x])), file=f1)
 
  f2 = open("./output/" + str(n) + "b.jpg.txt", 'w')

  for x in range(len(vec2)):
    print(' '.join(map(str, vec2[x])), file=f2)
  
