# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")

args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create, # video 5 terminal 1, failed on video 3
    "kcf": cv2.TrackerKCF_create, # done
    "boosting": cv2.TrackerBoosting_create, # failed on video 3, video 5 terminal 3
    "mil": cv2.TrackerMIL_create, # video 3 failed , video 5 terminal 2
    "tld": cv2.TrackerTLD_create, # video 1, 2 failed 
    "medianflow": cv2.TrackerMedianFlow_create, # done
    "mosse": cv2.TrackerMOSSE_create # done
}
# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# load the ground truth file
videoName = args["video"]
videoNameSplitted = videoName.split('.')
groundTruthFile = pd.read_csv("GroundTruth\\" + videoNameSplitted[0] + "\\gt.txt", ",", header=None)
# rename the columns so we know what are we working with
groundTruthFile['frameNumber'] = groundTruthFile[0]
groundTruthFile['objectId'] = groundTruthFile[1]
groundTruthFile['x'] = groundTruthFile[2]
groundTruthFile['y'] = groundTruthFile[3]
groundTruthFile['w'] = groundTruthFile[4]
groundTruthFile['h'] = groundTruthFile[5]
# get all rows that frameNumber = 1 - so it is possible to initialize the tracking of each object
firstFrameArray = groundTruthFile.loc[groundTruthFile['frameNumber'] == 1]

frameCounter = 0
initializedObjects = [False] * (max(groundTruthFile['objectId']) + 1) # max() is the number of objects that will be tracked in this video
idOfObject = []

f = open("ResultsTMP\\" + videoNameSplitted[0] + "_" + args.get("tracker") + ".txt", "w")

# number of objects currently tracked
cnt = 0
# threshold of how many objects to track
thresholdObjects = 50

# get video height and width
print(videoName)
if videoName == 'MOT20-01.avi':
    videoWidth = 1920
    videoHeight = 1080
if videoName == 'MOT20-02.avi':
    videoWidth = 1920
    videoHeight = 1080
if videoName == 'MOT20-03.avi':
    videoWidth = 1173
    videoHeight = 880 
if videoName == 'MOT20-05.avi':
    videoWidth = 1654
    videoHeight = 1080
    
print(str(videoWidth) + " " + str(videoHeight))

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frameCounter += 1
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    
   
    # loop over the bounding boxes and draw then on the frame
    objectCnt = 0
    for box in boxes:
        #print(str(box) + " " + str(idOfObject[cnt]))
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        f.write(str(frameCounter) + "," + str(idOfObject[objectCnt]) + "," + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "\n")
        objectCnt += 1

    # show the output frame
    print("Frame: " + str(frameCounter) + " , currently tracking " + str(cnt) + " objects with " + str(args.get("tracker")))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # check how many objects are currently tracked
    if (cnt + 1 <= thresholdObjects):    
        # get all objects that are seen in the current frameNumber 
        listOfObjectsInCurrentFrame = groundTruthFile.loc[groundTruthFile['frameNumber'] == frameCounter]
        # get all objects that are not already initialized in current frameNumber
        for x in listOfObjectsInCurrentFrame['objectId']: 
            if initializedObjects[x] == False: 

                row = listOfObjectsInCurrentFrame.loc[listOfObjectsInCurrentFrame['objectId'] == x]
                #row['w'] = row['w'].replace(1, 2)
                #row['h'] = row['h'].replace(1, 2)
                #row['w'] = row['w'].replace(0, 2)
                #row['h'] = row['h'].replace(0, 2)
                #if (int(row['x']) + int(row['w']) >= videoHeight): 
                #    if videoHeight - int(row['x']) >= 1:
                #        row['w'] = videoHeight - int(row['x'])
                #    else: 
                #        row['w'] = 1
                #if (int(row['y']) + int(row['h']) >= videoWidth): 
                #    if (videoWidth - int(row['y']) >= 1): 
                #        row['h'] = videoWidth - int(row['y'])
                #    else: 
                #        row['h'] = 1

                #print(str(frameCounter) + " " + str(box)) 
                box = (row['x'], row['y'], row['w'], row['h']) 
                try:
                    if cnt + 1 <= thresholdObjects:
                        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                        trackers.add(tracker, frame, box)
                        cnt += 1 # increase the number of tracked objects
                        print("Start tracking " + str(cnt))
                        # set the object to be initialized - ignore all future occurences
                        initializedObjects[x] = True
                        # initialize the tracker for the object
                        idOfObject.append(x)
                except: 
                    print("Error while initializing the tracker")
                    idOfObject.append(987654) # faulty box
                    

               
                
# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()
# close all windows
cv2.destroyAllWindows()
f.close()