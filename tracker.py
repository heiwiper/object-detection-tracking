import cv2
from random import randint
from configuration import TRACK_HIST_THRESHOLD, TRACKING_ALGO

class Tracker:
    def __init__(self):
        # Main counter which used for objects IDs
        self.count = 0

        # Unique ID of each object
        self.ids = []

        # Bounding box of each tracked object
        self.bboxes = []

        # How many frames each object wasn't detected/tracked
        self.history = []

        # After how many frames the object should be discarded
        self.historyThreshold = TRACK_HIST_THRESHOLD

        # Color of each object
        self.colors = []

        # List of each objects trajectory points
        self.trajectories = []

        # Tracker of each object
        self.trackers = []

        # File to save objects trajectories
        self.file = open('trajectories.txt', 'w')

    def addObject(self, bbox, frame):
        self.bboxes.append(bbox)
        # hls = (randint(0, 180), 127, 255)
        # color = cv2.cvtColor(np.uint8([[hls]]), cv2.COLOR_HLS2BGR)[0][0]
        # self.colors.append((int(color[0]),
        #                int(color[1]),
        #                int(color[2])))
        self.colors.append((randint(0, 255),
                            randint(0, 255),
                            randint(0, 255)))
        self.history.append(0)
        self.count += 1
        self.ids.append(self.count)
        newTrajectory = Trajectory()
        self.trajectories.append(newTrajectory)

        if TRACKING_ALGO == 1:
            newTracker = cv2.legacy.TrackerMOSSE_create()
        elif TRACKING_ALGO == 2:
            newTracker = cv2.TrackerKCF_create()
        elif TRACKING_ALGO == 3:
            newTracker = cv2.legacy.TrackerMIL_create()
        elif TRACKING_ALGO == 4:
            newTracker = cv2.legacy.TrackerCSRT_create()
        elif TRACKING_ALGO == 5:
            newTracker = cv2.TrackerGOTURN_create()
        elif TRACKING_ALGO == 6:
            newTracker = cv2.legacy.TrackerMedianFlow_create()
        newTracker.init(frame, (bbox[0], bbox[1], bbox[2], bbox[3]))
        self.trackers.append(newTracker)

    def removeObject(self, i):
        self.bboxes.pop(i)
        self.colors.pop(i)
        self.file.write("Object {}\n".format(self.ids[i]))
        for point in self.trajectories[i].points:
            self.file.write("\t\tin{:06}\t".format(point[2]))
            self.file.write("({}, {})\n".format(point[0], point[1]))
        self.file.write("\n".format(self.ids[i]))
        self.trajectories.pop(i)
        self.ids.pop(i)
        self.history.pop(i)
        self.trackers.pop(i)

    def update(self, frame, frameIndex, historyThreshold):
        for i, tracker in enumerate(self.trackers):
            success, self.bboxes[i] = tracker.update(frame)
            if success is False:
                self.history[i] += 1
                if self.history[i] > historyThreshold:
                    self.removeObject(i)
            else:
                x, y, w, h = self.bboxes[i]
                cx = int((x + x + w) // 2)
                cy = int((y + y + h) // 2)
                self.trajectories[i].points.append((cx, cy, frameIndex))

    def draw(self, frame):
        for i, bbox in enumerate(self.bboxes):
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2,
                            (self.colors[i][0],
                            self.colors[i][1],
                            self.colors[i][2]),
                            2, 1)
            for point in self.trajectories[i].points:
                cv2.circle(frame, (point[0], point[1]), radius=1,
                            color=(self.colors[i][0],
                                    self.colors[i][1],
                                    self.colors[i][2]),
                            thickness=-1)
            # cv2.rectangle(copy, p1, p2,
            #                 (colors[i][0],
            #                 colors[i][1],
            #                 colors[i][2]),
            #                 2, 1)


class Trajectory:
    def __init__(self):
        self.points = []
