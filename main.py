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

file = open('trajectories.txt', 'w')

count = 0
ids = []
tracking = []
history = []
colors = []
historyThreshold = 30
trajectories = []
trackers = []
class Trajectory:
    def __init__(self):
        self.points = []
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
    global count
    count += 1
    ids.append(count)
    newTrajectory = Trajectory()
    trajectories.append(newTrajectory)

    # newTracker = cv2.legacy.TrackerMedianFlow_create()  # Garbage
    # newTracker = cv2.legacy.TrackerCSRT_create()  # good but doesn't report failure
    newTracker = cv2.legacy.TrackerMOSSE_create()  # Very good
    # newTracker = cv2.legacy.TrackerMIL_create()  # okay, doesn't report failure
    # newTracker = cv2.TrackerGOTURN_create()
    # newTracker = cv2.TrackerKCF_create()  # Very good
    newTracker.init(frame, (bbox[0], bbox[1], bbox[2], bbox[3]))
    trackers.append(newTracker)


def removeTracker(i):
    tracking.pop(i)
    colors.pop(i)
    file.write("Object {}\n".format(ids[i]))
    for point in trajectories[i].points:
        file.write("\t\tin{:06}\t".format(point[2]))
        file.write("({}, {})\n".format(point[0], point[1]))
    file.write("\n".format(ids[i]))
    trajectories.pop(i)
    ids.pop(i)
    history.pop(i)
    trackers.pop(i)


#Main Function
def main():
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

    for frameIndex, frame in enumerate(frames):
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
                    rects = np.array([[x, y, x+w, y+h]
                                      for (x, y, w, h) in tracking+[bbox]])
                    pick = non_max_suppression(rects, probs=None,
                                               overlapThresh=0.65)

                    # Object is no longer detected
                    if pick.shape[0] == len(tracking) + 1:
                        addTracker(bbox, frame)




            # There already is one or multiple objects being tracked
            elif len(tracking) > 0:
                for i, bbox in enumerate(tracking):
                    # Use non_max_suppression algorithm to check if the tracked
                    # object is still detected, this helps in removing objects
                    # which are no longer visible
                    rects = np.array([[x, y, x+w, y+h]
                                      for (x, y, w, h) in detections+[bbox]])
                    pick = non_max_suppression(rects, probs=None,
                                               overlapThresh=0.65)

                    # Object is no longer detected
                    if pick.shape[0] == len(detections) + 1:
                        # Object was detected in at least the 5 last frames
                        if history[i] < historyThreshold:
                            history[i] += 3
                        # Object was not detected in 5 last frames, in this case
                        # we stop tracking the object
                        else:
                            removeTracker(i)
                            

            # Track every object in the tracking list
            for i, tracker in enumerate(trackers):
                success, tracking[i] = tracker.update(frame)
                if success is False:
                    history[i] += 1
                    if history[i] > historyThreshold:
                        removeTracker(i)
                else:
                    x, y, w, h = tracking[i]
                    cx = int((x + x + w) // 2)
                    cy = int((y + y + h) // 2)
                    trajectories[i].points.append((cx, cy, frameIndex))
            for i, bbox in enumerate(tracking):
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2,
                              (colors[i][0],
                               colors[i][1],
                               colors[i][2]),
                              2, 1)
                for point in trajectories[i].points:
                    cv2.circle(frame, (point[0], point[1]), radius=1,
                               color=(colors[i][0],
                                      colors[i][1],
                                      colors[i][2]),
                               thickness=-1)
                cv2.rectangle(copy, p1, p2,
                              (colors[i][0],
                               colors[i][1],
                               colors[i][2]),
                              2, 1)

        cv2.imshow('Frames', frame)
        cv2.imshow('Copy', copy)

        key = cv2.waitKey(30)
        if key == 27:
            break


if __name__ == "__main__":
    main()

file.close() 
print('Program Completed!')
