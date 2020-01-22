#!/usr/bin/env python

import cv2
import dlib
import numpy as np
import math
import sys
import os

def landmark_morph(numStart, numEnd, morphsFolder, morphOutputFolder):
    print("landmark morphs!")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./scripts/shape_predictor_68_face_landmarks.dat")

    # iterate through numbered files
    for n in range(numStart, numEnd):
        try:
            print(n)
            img1 = cv2.imread(morphsFolder + str(n) + "morph.png", 0)
            dets1 = detector(img1);
	 
            for k, d in enumerate(dets1):
                shape1  = predictor(img1, d)
	 
            vec = np.empty([68, 2], dtype = int)
            for b in range(68):
                vec[b][0] = (shape1.part(b).x)
                vec[b][1] = (shape1.part(b).y)
	   
            f1 = open(morphOutputFolder + str(n) + "morph.png.txt", 'w')
            for x in range(len(vec)):
                print(' '.join(map(str, vec[x])), file=f1)
        except Exception as e:
            print(str(e))
