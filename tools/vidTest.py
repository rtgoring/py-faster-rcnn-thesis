import cv2

cap = cv2.VideoCapture('test.avi')

while True:
    ret, im = cap.read()
    cv2.imshow('frame',im)
    cv2.waitKey(20)
    
