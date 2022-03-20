import os
import cv2
path = os.path.join(os.path.dirname(__file__), 'twitter_eu17b.jpg')
img = cv2.imread(path)

cv2.imshow('Image', img)
cv2.waitKey(0)
