from PIL import Image
import cv2
import numpy as np


frame = cv2.imread('blue_lexus_4.jpeg')

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Here we are defining range of bluecolor in HSV
# This creates a mask of blue coloured
# objects found in the frame.
mask = cv2.inRange(hsv, lower_blue, upper_blue)

cv2.imshow('mask', mask)
bluecnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
if len(bluecnts) > 0:
    blue_area = max(bluecnts, key=cv2.contourArea)
    (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
    cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)


    print(xg, yg, xg+wg, yg+hg)

cv2.rectangle(frame, (xg+68, yg+22),((xg+68 + wg-79),(yg+22 + hg-16)), (0, 255, 0), 2)

result = cv2.bitwise_and(frame, frame, mask=mask)

cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()






