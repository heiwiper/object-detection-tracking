import glob
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from tracker import Tracker

DIRECTORY_PATH = 'dataset/highway/input/'
filenames = [img for img in glob.glob(DIRECTORY_PATH+"*.jpg")]
filenames.sort()

frames = []
for frame in filenames:
    n = cv2.imread(frame)
    frames.append(n)


print("Finished reading images")

HIST_THRESHOLD = 30

tracker = Tracker()


#Main Function
def main():
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

    for frameIndex, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        copy = frame.copy()
        objects = car_cascade.detectMultiScale(gray,
                                            scaleFactor=1.05,
                                            minNeighbors=13)

        # Replace the detections list with detected objects
        detections = []
        for (x, y, w, h) in objects:
            cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
            detections.append([x, y, w, h])

        # One or multiple objects detected
        if len(detections) != 0:
            # No object is being tracked
            if len(tracker.bboxes) == 0:
                # Add the detected objects to be tracked
                for bbox in detections:
                    tracker.addObject(bbox, frame)

            elif len(tracker.bboxes) < len(detections):
                for index, bbox in enumerate(detections):
                    rects = np.array([[x, y, x+w, y+h]
                                      for (x, y, w, h) in tracker.bboxes+[bbox]])
                    pick = non_max_suppression(rects, probs=None,
                                               overlapThresh=0.65)

                    # Object is no longer detected
                    if pick.shape[0] == len(tracker.bboxes) + 1:
                        tracker.addObject(bbox, frame)

            # There already is one or multiple objects being tracked
            elif len(tracker.bboxes) > 0:
                for i, bbox in enumerate(tracker.bboxes):
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
                        if tracker.history[i] < HIST_THRESHOLD:
                            tracker.history[i] += 3
                        # Object was not detected in 5 last frames, in this case
                        # we stop tracking the object
                        else:
                            tracker.removeObject(i)

            # Track every object in the tracking list
            tracker.update(frame, frameIndex, HIST_THRESHOLD)

            # Draw bounding boxes and trajectories
            tracker.draw(frame)

        cv2.imshow('Frames', frame)
        cv2.imshow('Copy', copy)

        key = cv2.waitKey(30)
        if key == 27:
            break


if __name__ == "__main__":
    main()

tracker.file.close()
print('Program Completed!')
