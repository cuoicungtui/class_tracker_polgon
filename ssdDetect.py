import cv2
import numpy as np
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from random import randint


class Tracker_and_polygon():
    def __init__(self,width=1280, height=720):
        self.points = {}
        self.points['right'] = []
        self.points['left'] = []
        self.width = width
        self.height = height
        self.tracker_type = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    def load_points_form_json(self, path_json):
        try:
            with open(self.path_json) as json_file:
                data = json.load(json_file)
                self.points['left'] = data['left']
                self.points['right'] = data['right']
        except:
            print("Error: path json file is not exist")
    
    def handle_left_click(self, event, x, y, flags, points):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    def handle_point_click(self, event, x, y, flags, point_check):
        if event == cv2.EVENT_LBUTTONDOWN:
            point_check.append([x, y])

    # draw point check
    def draw_point_check(self,frame, point_check):
        for point in point_check:
            frame = cv2.circle( frame, (point[0], point[1]), 5, (65,33,1), -1)
    
    # draw polygon left and right
    def draw_polygon (self,frame):
        points = self.points
        for point in points['left']:
            frame = cv2.circle( frame, (point[0], point[1]), 3, (255,0,0), -1)
        
        for point in points['right']:
            frame = cv2.circle( frame, (point[0], point[1]), 3, (0,255,0), -1)

        frame = cv2.polylines(frame, [np.int32(points['left'])], False, (255,0, 0), thickness=1)
        frame = cv2.polylines(frame, [np.int32(points['right'])], False, (0,255, 0), thickness=1)
        return frame
    
    # draw tracker object
    def draw_tracker(self,frame,boxes,colors):
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        return frame
    # check point inside polygon left and right or outside polygon
    def isInside(self,points,centroid):
        polygon = Polygon(points)
        centroid = Point(centroid)
        # print(polygon.contains(centroid))
        return polygon.contains(centroid) 

    def centroid(self,bboxes):
        controids = []
        for box in bboxes:
            x, y, w, h = map(int, box)
            controids.append([x + w//2, y + h//2])
        return controids


    def write_points_title(self,points,frame):
        polygon = self.points
        for point in points:
            if self.isInside(polygon['left'], point):
                frame = self.alert(frame,"point in left",point)
            elif self.isInside(polygon['right'], point):
                frame = self.alert(frame,"point in right",point)
            else:
                frame = self.alert(frame,"point in outside polygon",point)
        return frame
    
    def alert(self, frame,text,point):
        frame = cv2.putText(frame, text, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame
    
    # function calculate distance between 2 point
    def distance(self,point1, point2):
        point1 =  Point(point1) 
        point2 =  Point(point2)
        return point1.distance(point2)

    def createTrackerByName(self,trackerType):

        if trackerType == self.tracker_type[0]:
            tracker = cv2.legacy.TrackerBoosting_create()
        elif trackerType == self.tracker_type[1]:
            tracker = cv2.legacy.TrackerMIL_create()
        elif trackerType == self.tracker_type[2]:
            tracker = cv2.legacy.TrackerKCF_create()
        elif trackerType == self.tracker_type[3]:
            tracker = cv2.legacy.TrackerTLD_create()
        elif trackerType == self.tracker_type[4]:
            tracker = cv2.legacy.TrackerMedianFlow_create()
        elif trackerType == self.tracker_type[5]:
            tracker = cv2.legacy.TrackerGOTURN_create()
        elif trackerType == self.tracker_type[6]:
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif trackerType == self.tracker_type[7]:
            tracker = cv2.legacy.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in self.tracker_type:
                print(t)
        return tracker

    def create_multi_track(self,frame):
        bboxes = []
        colors = []
        while True:
            # draw bounding boxes over objects
            # selectROI's default behaviour is to draw box starting from the center
            # when fromCenter is set to false, you can draw box starting from top left corner
            bbox = cv2.selectROI('MultiTracker', frame)
            bboxes.append(bbox)
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break
        return bboxes, colors
    
    def create_add_tracker(self,num_track_type,frame,bboxes):
        multiTracker = cv2.legacy.MultiTracker_create()
        for bbox in bboxes:
            multiTracker.add(self.createTrackerByName(self.tracker_type[num_track_type]), frame, bbox)
        return multiTracker
    
    # def draw_polygon_to_json(self,fram_name):
    #     cv2.setMouseCallback(fram_name, self.handle_point_click, point_check)
            # Press 'q' to exit the loop
    # if cv2.waitKey(1) & 0xFF == ord('t'):
    #     points['right'] = points['left']
    #     points['left'] = []