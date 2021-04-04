import glob
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from tracker import Tracker
from configuration import *

videos = ['backdoor', 'bungalows', 'highway', 'office', 'pedestrians', 'peopleInShade', 'PETS2006']
DIRECTORY_PATH = 'dataset/{}/input/'.format(videos[VIDEO-1])
filenames = [img for img in glob.glob(DIRECTORY_PATH+"*.jpg")]
filenames.sort()

frames = []
for frame in filenames:
    n = cv2.imread(frame)
    frames.append(n)


print("Finished reading images")

tracker = Tracker()


#Main Function
def main():
    if DETECTION_ALGO == 1:
        objectDetector = cv2.createBackgroundSubtractorMOG2(
            history=BG_HIST_THRESHOLD,
            varThreshold=BG_THRESHOLD)

    elif DETECTION_ALGO == 2:
        if HOG_MODEL == 1:
            objectDetector = cv2.CascadeClassifier('haarcascade_car.xml')
        elif HOG_MODEL == 2:
            objectDetector = cv2.HOGDescriptor()
            objectDetector.setSVMDetector(
                cv2.HOGDescriptor_getDefaultPeopleDetector())

    for frameIndex, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        copy = frame.copy()

        # Replace the detections list with detected objects
        detections = []
        if DETECTION_ALGO == 1:
            mask = objectDetector.apply(frame)
            _, mask = cv2.threshold(mask, 254, 255,
                                    cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
            rects = np.array([[x, y, x+w, y+h]
                              for (x, y, w, h) in [cv2.boundingRect(cnt)
                                                   for cnt in contours]])
            pick = non_max_suppression(rects, probs=None,
                                       overlapThresh=0.65)

            for (x1, y1, x2, y2) in pick:
                area = (x2-x1) * (y2-y1)
                if area > AREA_THRESHOLD:
                    cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    detections.append([x1, y1, x2-x1, y2-y1])
            cv2.imshow("Mask", mask)

        elif DETECTION_ALGO == 2:
            if HOG_MODEL == 1:
                objects = objectDetector.detectMultiScale(gray,
                                                          scaleFactor=SCALE_FACTOR,
                                                          minNeighbors=MIN_NEIGHBORS)
            elif HOG_MODEL == 2:
                (regions, _) = objectDetector.detectMultiScale(frame,
                                                               winStride=WIN_STRIDE,
                                                               padding=PADDING,
                                                               scale=SCALE_FACTOR)

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
                        if tracker.history[i] < TRACK_HIST_THRESHOLD:
                            tracker.history[i] += 3
                        # Object was not detected in 5 last frames, in this case
                        # we stop tracking the object
                        else:
                            tracker.removeObject(i)

            # Track every object in the tracking list
            tracker.update(frame, frameIndex, TRACK_HIST_THRESHOLD)

            # Draw bounding boxes and trajectories
            tracker.draw(frame)

        cv2.imshow('Detection', copy)
        cv2.imshow('Tracking', frame)

        key = cv2.waitKey(30)
        if key == 27:
            break


if __name__ == "__main__":
    main()

tracker.file.close()
print('Program Completed!')
