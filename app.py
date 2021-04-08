import os
import glob
import threading
# import threading
# from io import BytesIO
# import pickle
from os import walk
from io import BytesIO
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
# import time
import kivy
from kivy.app import App
from kivy.core.image import Image as CoreImage
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.graphics.texture import Texture
from kivy.config import Config
Config.set('kivy', 'exit_on_escape', '0')
from configuration import VIDEO, DETECTION_ALGO, TRACKING_ALGO, HOG_MODEL, BG_HIST_THRESHOLD, BG_THRESHOLD, AREA_THRESHOLD, PADDING, MIN_NEIGHBORS, WIN_STRIDE, TRACK_HIST_THRESHOLD, SCALE_FACTOR
from tracker import Tracker


kivy.require('2.0.0')

DATASETS_DIRS = ['backdoor', 'bungalows', 'highway', 'office',
            'pedestrians', 'peopleInShade', 'PETS2006']


class MainWindow(BoxLayout):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.appStarted = False
        self.playing = False
        self.videoChanged = False
        self.frames = []


    def reload(self):
        if self.appStarted is False or self.videoChanged is True:
            _, _, files = next(os.walk("dataset/{}/input".format(DATASETS_DIRS[VIDEO-1])))
            file_count = len(files)
            print("Files = {}".format(file_count))

            def load_images():
                content = ProgressBar(max = file_count)
                popup = Popup(title='Loading {} images...'.format(DATASETS_DIRS[VIDEO-1]),
                              content=content, size_hint=(0.4, 0.1),
                              auto_dismiss=False)
                popup.open()

                DIRECTORY_PATH = 'dataset/{}/input/'.format(DATASETS_DIRS[VIDEO-1])
                filenames = [img for img in glob.glob(DIRECTORY_PATH+"*.jpg")]
                filenames.sort()

                self.frames = []
                for frame in filenames:
                    n = cv2.imread(frame)
                    self.frames.append(n)
                    content.value += 1
                    popup.title = 'Loading {} images {}/{}'.format(DATASETS_DIRS[VIDEO-1], int(content.value),file_count)

                popup.dismiss()
                self.ids['video_spinner'].disabled = False
                self.ids['detection_spinner'].disabled = False
                self.ids['tracking_spinner'].disabled = False
                self.ids['play_button'].disabled = False
                self.ids['pause_button'].disabled = False
                if DETECTION_ALGO == 1:
                    self.ids['bg_sub_boxlayout'].disabled = False
                if DETECTION_ALGO == 2:
                    self.ids['hog_boxlayout'].disabled = False

            thread = threading.Thread(target=load_images)
            thread.start()
            self.appStarted = True
            self.videoChanged = False
        else:
            def start_tracking():
                tracker = Tracker(TRACKING_ALGO)
                self.log()
                print("[BBOXES] : {}".format(len(tracker.bboxes)))
                trace = True
                rectangle = True
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

                for frameIndex, frame in enumerate(self.frames):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    copy1 = frame.copy()
                    copy2 = frame.copy()

                    # Replace the detections list with detected objects
                    detections = []
                    if DETECTION_ALGO == 1:
                        mask = objectDetector.apply(copy1)
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
                                cv2.rectangle(copy2, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                detections.append([x1, y1, x2-x1, y2-y1])
                        cv2.imshow("Mask", mask)

                    elif DETECTION_ALGO == 2:
                        if HOG_MODEL == 1:
                            objects = objectDetector.detectMultiScale(gray,
                                                                    scaleFactor=SCALE_FACTOR,
                                                                    minNeighbors=MIN_NEIGHBORS)
                        elif HOG_MODEL == 2:
                            (objects, _) = objectDetector.detectMultiScale(copy1,
                                                                        winStride=WIN_STRIDE,
                                                                        padding=PADDING,
                                                                        scale=SCALE_FACTOR)

                        for (x, y, w, h) in objects:
                            cv2.rectangle(copy2, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            detections.append([x, y, w, h])

                    # One or multiple objects detected
                    if len(detections) != 0:
                        # No object is being tracked
                        if len(tracker.bboxes) == 0:
                            # Add the detected objects to be tracked
                            for bbox in detections:
                                tracker.addObject(bbox, copy1)

                        elif len(tracker.bboxes) < len(detections):
                            for index, bbox in enumerate(detections):
                                rects = np.array([[x, y, x+w, y+h]
                                                for (x, y, w, h) in tracker.bboxes+[bbox]])
                                pick = non_max_suppression(rects, probs=None,
                                                        overlapThresh=0.65)

                                # Object is no longer detected
                                if pick.shape[0] == len(tracker.bboxes) + 1:
                                    tracker.addObject(bbox, copy1)

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
                        tracker.update(copy1, frameIndex, TRACK_HIST_THRESHOLD)

                        # Draw bounding boxes and trajectories
                        tracker.draw(copy1, rectangle, trace)


                    message="Frame: {:03d}  Nbr objet: {:d}   [r]Rectangle: {:3}  [t]Trace: {:3}".format(frameIndex,
                          len(tracker.bboxes),
                          "ON" if rectangle else "OFF",
                          "ON" if trace else "OFF")
                
                    width = int(copy1.shape[1] * 2)
                    height = int(copy1.shape[0] * 2)

                    copy2 = cv2.resize(copy2, (width, height))
                    copy1 = cv2.resize(copy1, (width, height))

                    cv2.putText(copy1,
                                message,
                                (5, 15),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.1,
                                (100, 255, 70),
                                2)
                    cv2.imshow('Detection', copy2)
                    cv2.imshow('Tracking', copy1)

                    key = cv2.waitKey(30)
                    if key==ord('t'):
                        trace=not trace
                    if key==ord('r'):
                        rectangle=not rectangle
                    if key == 27:
                        tracker.file.close()
                        cv2.destroyAllWindows()
                        break
                
                tracker.file.close()
                cv2.destroyAllWindows()

            start_tracking()


    def switch_video(self):
        global VIDEO
        if self.ids['video_spinner'].text == "backdoor":
            VIDEO = 1
        elif self.ids['video_spinner'].text == "bungalows":
            VIDEO = 2
        elif self.ids['video_spinner'].text == "highway":
            VIDEO = 3
        elif self.ids['video_spinner'].text == "office":
            VIDEO = 4
        elif self.ids['video_spinner'].text == "pedestrians":
            VIDEO = 5
        elif self.ids['video_spinner'].text == "PETS2006":
            VIDEO = 6
        print(self.ids['video_spinner'].text)
        self.videoChanged = True
        self.reload()

    def switch_detection_algo(self):
        global DETECTION_ALGO
        if self.ids['detection_spinner'].text == "Background\nsubstraction":
            DETECTION_ALGO = 1
            self.ids['bg_sub_boxlayout'].disabled = False
            self.ids['hog_boxlayout'].disabled = True
        elif self.ids['detection_spinner'].text == "Haarcascade":
            DETECTION_ALGO = 2
            self.ids['hog_boxlayout'].disabled = False
            self.ids['bg_sub_boxlayout'].disabled = True
        print(self.ids['detection_spinner'].text)

    def switch_tracking_algo(self):
        global TRACKING_ALGO
        if self.ids['tracking_spinner'].text == "MOSSE":
            TRACKING_ALGO = 1
        elif self.ids['tracking_spinner'].text == "KFC":
            TRACKING_ALGO = 2
        elif self.ids['tracking_spinner'].text == "MIL":
            TRACKING_ALGO = 3
        elif self.ids['tracking_spinner'].text == "CSRT":
            TRACKING_ALGO = 4
        elif self.ids['tracking_spinner'].text == "GOTURN":
            TRACKING_ALGO = 5
        elif self.ids['tracking_spinner'].text == "Median Flow":
            TRACKING_ALGO = 6
        print("{} {}".format(self.ids['tracking_spinner'].text, TRACKING_ALGO))

    def switch_model(self, model):
        global HOG_MODEL
        HOG_MODEL = model

    def update_bg_hist_thresh(self):
        global BG_HIST_THRESHOLD
        BG_HIST_THRESHOLD = int(self.ids['bg_hist_thresh_slider'].value)
            
    def update_bg_thresh(self):
        global BG_THRESHOLD
        BG_THRESHOLD = int(self.ids['bg_thresh_slider'].value)

    def update_area_thresh(self):
        global AREA_THRESHOLD
        AREA_THRESHOLD = int(self.ids['area_thresh_slider'].value)

    def update_scale_factor(self):
        global SCALE_FACTOR
        SCALE_FACTOR = self.ids['scale_factor_slider'].value

    def update_min_neighbors(self):
        global MIN_NEIGHBORS
        MIN_NEIGHBORS = int(self.ids['min_neighbors_slider'].value)
        
    def update_padding(self):
        global PADDING
        temp = int(self.ids['padding_slider'].value)
        PADDING = (temp, temp)
        
    def update_win_stride(self):
        global WIN_STRIDE
        temp = int(self.ids['win_stride_slider'].value)
        WIN_STRIDE = (temp, temp)

    def log(self):
        print("Configuration---------------------")
        if DETECTION_ALGO == 1:
            print("[Detection] : Background substraction")
            print("[Bg sub threshold] : {}".format(BG_THRESHOLD))
            print("[Bg sub hist threshold] : {}".format(BG_HIST_THRESHOLD))
            print("[Area threshold] : {}".format(AREA_THRESHOLD))
        elif DETECTION_ALGO == 2:
            print("[Detection] : HOG")
            if HOG_MODEL == 1:
                print("[HOG Model] : Cars")
            elif HOG_MODEL == 2:
                print("[HOG Model] : Pedestrians")
            print("[Window Stride] : {}".format(WIN_STRIDE))
            print("[Padding] : {}".format(PADDING))
            print("[Min neighbors] : {}".format(MIN_NEIGHBORS))
            print("[Scale factor] : {}".format(SCALE_FACTOR))
        if DETECTION_ALGO == 1:
            print("[Tracking] : MOSSE")
        elif DETECTION_ALGO == 2:
            print("[Tracking] : KFC")
        elif DETECTION_ALGO == 3:
            print("[Tracking] : MIL")
        elif DETECTION_ALGO == 4:
            print("[Tracking] : CSRT")
        elif DETECTION_ALGO == 5:
            print("[Tracking] : GOTURN")
        elif DETECTION_ALGO == 6:
            print("[Tracking] : Median Flow")

        
class TrackingApp(App):
    def build(self):
        # videos = ['backdoor', 'bungalows', 'highway', 'office', 'pedestrians', 'peopleInShade', 'PETS2006']
        # DIRECTORY_PATH = 'dataset/{}/input/'.format(videos[VIDEO-1])
        # filenames = [img for img in glob.glob(DIRECTORY_PATH+"*.jpg")]
        # filenames.sort()

        # frames = []
        # for frame in filenames:
        #     n = cv2.imread(frame)
        #     frames.append(n)


            
        window = MainWindow()



        
        return window


if (__name__ == '__main__'):
    TrackingApp().run()
