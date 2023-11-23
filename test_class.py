import cv2
import numpy as np
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from random import randint
from ssdDetect import Tracker_and_polygon

tracker_polygons = Tracker_and_polygon(1280, 720)


path_json = 'polygon.json'
video_path = '../examples/video_data/roadTraffic.mp4'
count =1
num_frame_to_detect = 20
point_check = []

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

tracker_polygons.load_points_form_json(path_json)
bboxes, colors = tracker_polygons.create_multi_track(frame)

print('Selected bounding boxes {}'.format(bboxes))

multiTracker = tracker_polygons.create_add_tracker(4,frame, bboxes)

while cap.isOpened():
    count+=1
    ret, frame = cap.read()
    if not ret:
        break
    if count ==num_frame_to_detect+1:
        count =1
        if cv2.waitKey(0) & 0xFF == 113:
            print('Selected bounding boxes new')
            bboxes, colors = tracker_polygons.create_multi_track(frame)         
            multiTracker = tracker_polygons.create_add_tracker(4,frame, bboxes)

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # Ve ploygon
    frame = tracker_polygons.draw_polygon(frame)

    # draw tracked objects
    frame = tracker_polygons.draw_tracker(frame, bboxes, colors)

    # after 50 frame check select box with polygon and write to title and 
    if count % num_frame_to_detect ==0:
        controids = tracker_polygons.centroid(boxes)
        tracker_polygons.draw_point_check(frame, controids)
        frame = tracker_polygons.write_points_title(controids, frame)

    # show frame
    cv2.imshow('MultiTracker', frame)
    # wait for 5 sec
    if count % num_frame_to_detect==0:
        cv2.waitKey(1000)
    
       # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break