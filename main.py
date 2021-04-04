import glob
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from random import randint

DIRECTORY_PATH = 'dataset/highway/input/'
filenames = [img for img in glob.glob(DIRECTORY_PATH+"*.jpg")]
filenames.sort()

frames = []
for frame in filenames:
    n = cv2.imread(frame)
    frames.append(n)


print("Finished reading images")

tracking = []
history = []
colors = []
historyThreshold = 30

trackers = []

def addTracker(bbox, frame):
    tracking.append(bbox)
    # hls = (randint(0, 180), 127, 255)
    # color = cv2.cvtColor(np.uint8([[hls]]), cv2.COLOR_HLS2BGR)[0][0]
    # colors.append((int(color[0]),
    #                int(color[1]),
    #                int(color[2])))
    colors.append((randint(0, 255),
                   randint(0, 255),
                   randint(0, 255)))
    history.append(0)

    # newTracker = cv2.legacy.TrackerMedianFlow_create() # Really bad
    # newTracker = cv2.legacy.TrackerCSRT_create() # good but doesn't report failure
    newTracker = cv2.legacy.TrackerMOSSE_create() # Very good
    # newTracker = cv2.legacy.TrackerMIL_create() # okay, doesn't report failure
    # newTracker = cv2.TrackerGOTURN_create() #
    # newTracker = cv2.TrackerKCF_create() # Very good
    success = newTracker.init(frame, (bbox[0], bbox[1], bbox[2], bbox[3]))
    trackers.append(newTracker)

                    
#Main Function
def main():
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        copy = frame.copy()
        cars = car_cascade.detectMultiScale(gray,
                                            scaleFactor=1.05,
                                            minNeighbors=13)

        # Replace the detections list with detected objects
        detections = []
        for (x, y, w, h) in cars:
            cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
            detections.append([x, y, w, h])

        # One or multiple objects detected
        if len(detections) != 0:
            # No object is being tracked
            if len(tracking) == 0:
                # Add the detected objects to be tracked
                for bbox in detections:
                    addTracker(bbox, frame)

            elif len(tracking) < len(detections):
                for index, bbox in enumerate(detections):
                    rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in tracking+[bbox]])
                    pick = non_max_suppression(rects, probs=None,
                                            overlapThresh=0.65)

                    # Object is no longer detected
                    if pick.shape[0] == len(tracking) + 1:
                        addTracker(bbox, frame)

            # Track every object in the tracking list
            for i, tracker in enumerate(trackers):
                success, tracking[i] = tracker.update(frame)
                if success is False:
                    history[i] += 1
                    if history[i] > historyThreshold:
                        tracking.pop(i)
                        colors.pop(i)
                        history.pop(i)
                        trackers.pop(i)
            for i, bbox in enumerate(tracking):
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2,
                                (colors[i][0], colors[i][1], colors[i][2]),
                                2, 1)
                cv2.rectangle(copy, p1, p2,
                                (colors[i][0], colors[i][1], colors[i][2]),
                                2, 1)

        # print(detections)
        cv2.imshow('Frames', frame)
        cv2.imshow('Copy', copy)

        key = cv2.waitKey(30)
        if key == 27:
            break
if __name__ == "__main__":
    main()

print('Program Completed!')
