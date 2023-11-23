import cv2
import numpy as np
from imutils.video import VideoStream
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from random import randint

# function draw polygon

points = {}
points['right'] = []
points['left'] = []

# click left mouse to add point for polygon of dict points left and right 
def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

# click left mouse to add point for list point_check 
def handle_point_click(event, x, y, flags, point_check):
    if event == cv2.EVENT_LBUTTONDOWN:
        point_check.append([x, y])

# draw point check
def draw_point_check(frame, point_check):
    for point in point_check:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (65,33,1), -1)

# draw polygon left and right
def draw_polygon (frame, points):
    for point in points['left']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (255,0,0), -1)
    
    for point in points['right']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (0,255,0), -1)

    frame = cv2.polylines(frame, [np.int32(points['left'])], False, (255,0, 0), thickness=1)
    frame = cv2.polylines(frame, [np.int32(points['right'])], False, (0,255, 0), thickness=1)
    return frame

# load points of polygon from json file
def load_points_form_json(path_json):
    with open(path_json) as json_file:
        data = json.load(json_file)
        points['left'] = data['left']
        points['right'] = data['right']
    return points

# check point inside polygon left and right or outside polygon
def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    # print(polygon.contains(centroid))
    return polygon.contains(centroid)

# write title for point in point_check when check point inside polygon left and right or outside polygon
def write_points_title(points, polygon,frame):
    for point in points:
        if isInside(polygon['left'], point):
            frame = cv2.putText(frame, "point in left", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif isInside(polygon['right'], point):
            frame = cv2.putText(frame, "point in right", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            frame = cv2.putText(frame, "point in outside polygon", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame

# Create a tracker based on tracker name
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):

    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
    return tracker

# function centroid calculate centroid of boxs
def centroid(bboxes):
    controids = []
    for box in bboxes:
        x, y, w, h = map(int, box)
        controids.append([x + w//2, y + h//2])
    return controids

# function calculate distance between 2 point
def distance(point1, point2):
    point1 =  Point(point1) 
    point2 =  Point(point2)
    return point1.distance(point2)

# Open a video capture object
video_path = '../examples/video_data/roadTraffic.mp4'
cap = cv2.VideoCapture(video_path)

# load points from json file
points_json = load_points_form_json('polygon.json')

# points check
point_check = []


## Select boxes
bboxes = []
colors = []

# Read first frame
ret, frame = cap.read()

# Create MultiTracker object and quit for 'q'
# while True:
#     # draw bounding boxes over objects
#     # selectROI's default behaviour is to draw box starting from the center
#     # when fromCenter is set to false, you can draw box starting from top left corner
#     bbox = cv2.selectROI('MultiTracker', frame)
#     bboxes.append(bbox)
#     colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
#     print("Press q to quit selecting boxes and start tracking")
#     print("Press any other key to select next object")
#     k = cv2.waitKey(0) & 0xFF
#     if (k == 113):  # q is pressed
#         break

def create_multi_tracker(frame):
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

bboxes, colors = create_multi_tracker(frame)

print('Selected bounding boxes {}'.format(bboxes))

trackerType = trackerTypes[4]  
 
# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

count =1
num_frame_to_detect = 20
while cap.isOpened():
    count+=1
    ret, frame = cap.read()
    if not ret:
        break

    if count ==num_frame_to_detect+1:
        count =1
        if cv2.waitKey(0) & 0xFF == 113:
            multiTracker = cv2.legacy.MultiTracker_create()
            print('Selected bounding boxes new')
            bboxes, colors = create_multi_tracker(frame)
            for bbox in bboxes:
                multiTracker.add(createTrackerByName(trackerType), frame, bbox)


    # Ve ploygon
    frame = draw_polygon(frame, points_json)
    
    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    
    # after 50 frame check select box with polygon and write to title and 
    if count % num_frame_to_detect ==0:
        controids = centroid(boxes)
        draw_point_check(frame, controids)
        frame = write_points_title(controids, frame)
        



    # show frame
    cv2.imshow('MultiTracker', frame)

    # wait for 5 sec
    if count % num_frame_to_detect==0:
        cv2.waitKey(1000)


    # cv2.setMouseCallback('Intrusion Warning', handle_point_click, point_check)

    # Press 'q' to exit the loop
    # if cv2.waitKey(1) & 0xFF == ord('t'):
    #     points['right'] = points['left']
    #     points['left'] = []

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print("points left: ", points['left'])
# print("\n")
# print("points right: ", points['right'])
# print("\n")
# print("points check: ", point_check)

# # Specify the file path where you want to save the JSON data
# file_path = "polygon.json"

# # Write the dictionary to a JSON file
# with open(file_path, 'w') as json_file:
#     json.dump(points, json_file)
