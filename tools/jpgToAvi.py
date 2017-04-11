import numpy as np
import cv2
import os

allImages = []
fileDirectory = 'output/boaty/49/'
#fileDirectory = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437508353_14466389'
for f in os.listdir(fileDirectory):
    if f.endswith('.jpg') or f.endswith('jpeg'):
        allImages.append(f)

allImages = sorted(allImages)
currentImagePos = 0
imageNum = 0



# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('demo5'+'.avi',fourcc, 15.0, (1920,1200))
##out = cv2.VideoWriter('Cam1-GBY.avi',fourcc, 15.0, (2000,1000))
for image in allImages:
    print image
    frame = cv2.imread(os.path.join(fileDirectory,image))

    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print 'done'
# Release everything if job is finished
out.release()
print 'released'
cv2.destroyAllWindows()
print 'destroyed'
