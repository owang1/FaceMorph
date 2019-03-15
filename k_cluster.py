#! usr/bin/env python

# Source code from https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
  numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
  (hist, _) = np.histogram(clt.labels_, bins=numLabels)

  hist = hist.astype("float")
  hist /= hist.sum()

  return hist

def plot_colors2(hist, centroids):
  bar = np.zeros((50, 300, 3), dtype="uint8")
  startX = 0

  for(percent, color) in zip(hist, centroids):
    # plot the relative percentage of each cluster
    endX = startX + (percent * 300)
    cv2.rectangele(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
 
    startX = endX
  return bar

img = cv2.imread("/morphs/3_morph.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((img.shape[0] * img.shape[1], 3)) 
clt = KMeans(n_clusters=3) 
clt.fit(img)

hist = find_histogram(clt)
bar = plot_colors2(hist, clt.cluters_centers_)
plt.axis("off")
plot.imshow(bar)
plot.show()
