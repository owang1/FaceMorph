#! usr/bin/env python

# Program that replaces black background with skin colored background
import numpy as np
import cv2

img = cv2.imread('./morphs/3_morph.jpg')
# background = cv2.imread('./pairs/3a.jpg')

# Range of blacks
lowerBlack = np.array([0,0,0], dtype = "uint8")
upperBlack = np.array([20,20,20], dtype = "uint8")

# Target color (skin)
color = [206, 211, 255]

# Create mask using inRange
mask = cv2.inRange(img, lowerBlack, upperBlack)

# Replace black with skin color
img[mask>0]=color

# Show and save image
cv2.imshow('img', img)
cv2.imwrite('test.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
