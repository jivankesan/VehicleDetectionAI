import cv2
import sys
import numpy as np
import matplotlib as plt


def draw_bounding_box(contours, image, number_of_boxes=1):
    # Call our function to get the list of contour areas
    cnt_area = contour_area(contours)

    # Loop through each contour of our image
    for i in range(0, len(contours), 1):
        cnt = contours[i]

        # Only draw the the largest number of boxes
        if (cv2.contourArea(cnt) > cnt_area[number_of_boxes]):
            # Use OpenCV boundingRect function to get the details of the contour
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw the bounding box
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image


def contour_area(contours):
    # create an empty list
    cnt_area = []

    # loop through all the contours
    for i in range(0, len(contours), 1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt_area.append(cv2.contourArea(contours[i]))

    # Sort our list of contour areas in descending order
    list.sort(cnt_area, reverse=True)
    return cnt_area

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    # Set up tracker.

    tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture("blue_lexus_1.mp4")
    # video = cv2.VideoCapture(0) # for using CAM

    # if using a video,

    # Reading first frame
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    _, frame = video.read()
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.imshow('mask', mask)

    # The bitwise and of the frame and mask is done so
    # that only the blue coloured objects are highlighted
    # and stored in res
    res = cv2.bitwise_and(frame, frame, mask=mask)


    # initializing the bounding box
    bbox = (xg, yg, xg+wg, yg+hg)

    # manual select bounding box
   # bbox = cv2.selectROI(frame, False)

    # initializing the tracker
    ok = tracker.init(frame, bbox)

    print(bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

         # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        # Display result
        cv2.imshow("Tracking", frame)
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
            break
    video.release()
    print(bbox)
    cv2.destroyAllWindows()

