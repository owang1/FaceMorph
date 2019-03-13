#!/usr/bin/env python

import cv2
import dlib
import numpy as np
import math
import sys


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread("./morphs/3_morph.jpg", 0)

dets = detector(img)

for k, d in enumerate(dets):
  shape = predictor(img, d)

vec = np.empty([68, 2], dtype = int)
for b in range(68):
  vec[b][0] = shape.part(b).x
  vec[b][1] = shape.part(b).y

f = open("./output/3_morph.jpg.txt", 'w')

for x in range(len(vec)):
  print(' '.join(map(str, vec[x])), file=f)
