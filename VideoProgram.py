import cv2
import sys
import numpy as np

if __name__ == '__main__':

    # Set up tracker.

    tracker = cv2.TrackerCSRT_create()
    tracker2 = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture(0)

    # Reading first frame
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    ok, frame = video.read()

    _, frame = video.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colour = input()
    if colour == 'blue':
        u = [110,50,50]
        l = [130,255,255]
    elif colour =='green':
        u = [36, 50, 70]
        l = [89, 255, 255]
    elif colour == 'white':
        u = [0,0,231]
        l = [180,18,255]
    else:
        u = [0,0,0]
        l = [180,255,30]
    lower = np.array(u)
    upper = np.array(l)

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow('mask', mask)
    bluecnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(bluecnts) > 0:
        blue_area = max(bluecnts, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

        print(xg, yg, wg, hg)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('blue', result)

    # initializing the bounding box
    bbox = (xg,yg,wg,hg)
    bbox2 = (xg+68, yg+22, wg-79, hg-16)

    initialx = xg

    # initializing the tracker
    ok = tracker.init(frame, bbox)
    ok = tracker2.init(frame, bbox2)

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
        ok, bbox2 = tracker2.update(frame)

        # DistanceCalc
        distance = (bbox[0] - initialx)/4.3

        # Update tracker
        ok, bbox = tracker.update(frame)
        ok, bbox2 = tracker2.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

            p3 = (int(bbox2[0]), int(bbox2[1]))
            p4 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p3, p4, (0, 255, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        # Display distance on frame
        cv2.putText(frame, "distance : " + str(int(distance)) + " cm", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        # Display result
        cv2.imshow("Tracking", frame)
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
            break

    video.release()
    print(bbox)
    print(distance)
    cv2.destroyAllWindows()
