import cv2
import numpy as np
import glob

frameSize = (1920, 734)

out = cv2.VideoWriter('mot20-08.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)

for filename in glob.glob('MOT20\\test\\MOT20-08\\img1\\*.jpg'):
    print(filename)
    img = cv2.imread(filename)
    out.write(img)

out.release()