import cv2 as cv2
import os
import numpy as np


ret,thresh = cv2.threshold(gray,127,255,0)

# find the contours
contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# take the first contour
cnt = contours[0]

# compute the bounding rectangle of the contour
x,y,w,h = cv2.boundingRect(cnt)

# draw contour
img = cv2.drawContours(img,[cnt],0,(0,255,255),2)

# draw the bounding rectangle
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# display the image with bounding rectangle drawn on it
cv2.imshow("Bounding Rectangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()