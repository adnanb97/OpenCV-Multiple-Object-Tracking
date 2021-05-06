import pandas as pd
import math 



# /////////////////////////////////////////
# /      Begin of metric functions       /
# ////////////////////////////////////////

# first function to return tracking benchmark
def center_distance(boxA, boxB):
    # determine the (x, y)-coordinates of the centers of rectangle
    centerAx = boxA[0] + boxA[2] / 2
    centerAy = boxA[1] + boxA[3] / 2
    centerBx = boxB[0] + boxB[2] / 2
    centerBy = boxB[1] + boxB[3] / 2
    xKvadrat = (centerAx - centerBx) * (centerAx - centerBx)
    yKvadrat = (centerAy - centerBy) * (centerAy - centerBy) 
    # compute the distance
    distance = math.sqrt(xKvadrat + yKvadrat)
    
 
    # return the distance between centers
    return distance

# second function to return tracking benchmark
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

# /////////////////////////////////////////
# /        End of metric functions       /
# ////////////////////////////////////////

# function that takes three arguments: video name, name of tracker and object id
# returns the CD and IoU values when comparing the tracker (trackerName) with the ground truth, on video (videoName) for a specified object id
def process(videoName, trackerName, desiredObjectid):

    # read the ground truth file for the video provided
    groundTruth = pd.read_csv('MOT20/train/' + videoName + '/gt/gt.txt', ',', header=None)
    groundTruth = groundTruth.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h', 
        6: 'confidence', 
        7: 'typeOfObject'
    })

    # read the results file we got
    df = pd.read_csv('Results/' + videoName + '_' + trackerName + '.txt', ',', header=None)
    df = df.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h'
    })

    # which object are we analyzing?
    queryObjectID = desiredObjectid

    objectID1_res = df.loc[df['objectID'] == queryObjectID]
    objectID1_gt = groundTruth.loc[groundTruth['objectID'] == queryObjectID]
    
    if (len(objectID1_gt) == 0): 
        return
    # drop the first frame in gt file - the tracker is initialized with these values, so the results start from second frame
    #print(objectID1_gt)
    #objectID1_gt = objectID1_gt.drop([0], axis=0)

    # check if the lengths are the same - did the tracker detect dissapearing of the object?
    if (len(objectID1_gt) < len(objectID1_res)):
        print(trackerName + " did not detect disappearing of the object in frame " + str(len(objectID1_gt))) 

    # get the upper bound length (the minimum of two values) to compare frame by frame
    upperBound = min(len(objectID1_gt), len(objectID1_res))
    iou = 0
    cd = 0
    numOfAnalyzed = 0
    for i in range(upperBound - 1): 
        # check if the frame number is the same (i + 1 on GT because first row contains frame 1 which is not shown in the results - it is only for initialization of the tracker)
        if (objectID1_gt.iloc[i + 1]['frameNumber'] == objectID1_res.iloc[i]['frameNumber']):
            # take the current frames bounding boxes to compute metrics IoU and CD
            boxA = (objectID1_res.iloc[i]['x'], objectID1_res.iloc[i]['y'], objectID1_res.iloc[i]['x'] + objectID1_res.iloc[i]['w'], objectID1_res.iloc[i]['y'] + objectID1_res.iloc[i]['h'])
            boxB = (int(objectID1_gt.iloc[i + 1]['x']), int(objectID1_gt.iloc[i + 1]['y']), int(objectID1_gt.iloc[i + 1]['x']) + int(objectID1_gt.iloc[i + 1]['w']), int(objectID1_gt.iloc[i + 1]['y']) + int(objectID1_gt.iloc[i + 1]['h']))
            # call the predefined metric function
            intersectionOverUnion = intersection_over_union(boxA, boxB)
            centerDistance = center_distance(boxA, boxB)
            
            # then compute the values
            
            cd += centerDistance 
            iou += intersectionOverUnion
            numOfAnalyzed += 1
    if (numOfAnalyzed != 0): 
        iou /= numOfAnalyzed
        cd /= numOfAnalyzed
    else: 
        print("Object with ID " + str(queryObjectID) + " was not tracked with " + trackerName + " in video " + videoName + ".")

    #print("Average IoU for " + trackerName + " on video " + videoName + " for objectID " + str(queryObjectID) + " is = " + str(iou))
    #print("Average CD for " + trackerName + " on video " + videoName + " for objectID " + str(queryObjectID) + " is = " + str(cd))
    return (iou, cd)

# function that takes two arguments: name of video and name of the tracker
# function returns an array of object ids tracked in that video with that tracker
def getTrackedObjectIds(videoName, trackerName):
    df = pd.read_csv('Results/' + videoName + '_' + trackerName + '.txt', ',', header=None)
    df = df.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h'
    })

    listOfIds = []

    df = df['objectID']
    for x in df: 
        # check if current ID is a new one
        if (x not in listOfIds) and x != 987654: 
            # double check - not to go over 50 objects
            if (len(listOfIds) + 1 <= 50):
                listOfIds.append(x)

    return listOfIds

# function that takes two arguments - list of objects tracked in a video and video name
# function outputs pair object of structure (objectID, objectType)
def getObjectTypes(listOfIds, videoName):
    returnArray = []

    groundTruth = pd.read_csv('MOT20/train/' + videoName + '/gt/gt.txt', ',', header=None)
    groundTruth = groundTruth.rename(columns={
        0: 'frameNumber', 
        1: 'objectID', 
        2: 'x', 
        3: 'y', 
        4: 'w', 
        5: 'h', 
        6: 'confidence', 
        7: 'typeOfObject'
    })

    for objectID in listOfIds: 
        # find the row where objectID is located
        object_type = groundTruth.loc[groundTruth['objectID'] == objectID]['typeOfObject']
        #print(str(object_type.iloc[0]))
        returnArray.append((objectID , object_type.iloc[0]))
    
    return returnArray
        

videoNames = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
trackerNames = ['csrt', 'kcf', 'boosting', 'mil', 'tld', 'medianflow', 'mosse']

objectTypes = []
for videoName in videoNames: 
    for trackerName in trackerNames: 
        listOfIds = getTrackedObjectIds(videoName, trackerName)
        currentObjectArray = getObjectTypes(listOfIds, videoName)
        print("Processing " + videoName + " " + trackerName)
        for x in currentObjectArray: 
            if x not in objectTypes: 
                objectTypes.append(x)

f = open("ResultsProcessed/objectTypes.txt", "w")
for x in objectTypes: 
    f.write(str(x[0]) + " " + str(x[1]) + "\n")
f.close()

#listOfIds = getTrackedObjectIds(videoNames[0], trackerNames[0])
#print(listOfIds)
#process(videoNames[0], 'mosse', 1)
#for videoName in videoNames: 

#videoName = videoNames[3]
#for trackerName in trackerNames:
#    listOfIds = getTrackedObjectIds(videoName, trackerName)
    # write the results to a file
#    f = open("ResultsProcessed/" + videoName + "_" + trackerName + ".txt", "w")
    #print(videoName + " " + trackerName + " " + str(listOfIds))
#    objectNumber = 0
    #print("Objects tracked for video " + videoName + " and tracker " + trackerName + " = " + str(listOfIds))
#    for oneObject in listOfIds: 
#        objectNumber += 1
#        print("Analyzing object " + str(objectNumber) + " on video " + videoName + " with tracker " + trackerName)
#        (iou, cd) = process(videoName, trackerName, oneObject)
#        f.write(str(oneObject) + " " + str(iou) + " " + str(cd) + "\n")
    
#    f.close()