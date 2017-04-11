#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

INPUT_MODE = 0 #0 = images, 1 = Camera, 2 = ScreenCapture
GUI = 1
RECORD = 1
PRINT_STATS = 0


import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time

found_objects = []

class camera: #cameraProperties
    def __init__ (self, camera_model,frame_rate, fov_width, fov_height, focal_length, camera_direction):
        self.cameraModel = camera_model
        self.FRAME_RATE = frame_rate
        self.FOV_WIDTH = fov_width
        self.FOV_HEGIHT = fov_height
        self.FOCAL_LENGTH = focal_length
        self.CAMERA_DIRECTION = camera_direction # 1=Foward, 2=Down


class detectedObject:
    def __init__(self, name):
        self.name = name
        self.px_location = ((None,None),(None,None))
        #self.average_px_location =((None,None),(None, None))
        self.prev_px_location = ((0,0),(0,0))
        self.inFrame = None
        self.width = None#Calculated - Not Shown
        self.height = None#Calculated - Not Shown
        self.rl_location = (None,None,None)#Not Calculated - Not Shown
        self.size_array = np.zeros([5]) #Not Calculated - Not Shown
        self.size = None#Calculated - Not Shown
        self.distance = None#Not Calculated - Not Shown
        self.average_size = None#Not Calculated - Not Shown
        self.life = 1#Calculated -  Shown
        self.confidence = None #Calculated - Not Shown
        self.area = None #Calculated - Not Shown
        self.objectCenter = None #Calculated - Not Shown
        self.time_not_seen = 0
        self.average_location_bin = 3
        
        self.x1_array = np.zeros([self.average_location_bin])
        self.y1_array = np.zeros([self.average_location_bin])
        self.x2_array = np.zeros([self.average_location_bin])
        self.y2_array = np.zeros([self.average_location_bin])

        self.x1_average = None
        self.y1_average = None
        self.x2_average = None
        self.y2_average = None

    def average_location(self, new_x1, new_y1, new_x2, new_y2):
        self.x1_array = np.roll(self.x1_array,1)
        self.y1_array = np.roll(self.y1_array,1)
        self.x2_array = np.roll(self.x2_array,1)
        self.y2_array = np.roll(self.y2_array,1)
        
        self.x1_array[self.average_location_bin-1] = new_x1
        self.y1_array[self.average_location_bin-1] = new_y1
        self.x2_array[self.average_location_bin-1] = new_x2
        self.y2_array[self.average_location_bin-1] = new_y2

        self.x1_average = int(np.average(self.x1_array))
        self.y1_average = int(np.average(self.y1_array))
        self.x2_average = int(np.average(self.x2_array))
        self.y2_average = int(np.average(self.y2_array))


    def update_name(self, new_name):
        self.name = new_name

    def update_px_location(self, new_px_location):
        self.px_location = new_px_location

    def update_rl_location(self, new_rl_location):
        self.rl_location = new_rl_location

    def update_size(self, new_size):
        self.size = new_size

    def update_distance(self, new_distance):
        self.distance = new_distance

    def shift_size(self, new_size):
        self.size_array = np.roll(self.size_array,1)
        self.size_array[4] = new_size
        self.update_avg_size()
        
    def update_avg_size(self):
        self.average_size = np.average(self.size_array)
        #print self.average_size
        
    def print_sizes(self):
        print self.size_array

    def increment_life(self):
        self.life = self.life + 1


def distance_to_camera_mm(focal_mm,real_height_mm,image_height_px,object_height_px,sensor_height_mm):
    #http://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo 
    return (focal_mm*real_height_mm*image_height_px)/(object_height_px * sensor_height_mm)


def distance_to_camera(knownWidth, cameraFocalLength, perceivedWidth):
    if perceivedWidth:
        return (knownWidth * cameraFocalLength) / perceivedWidth
    else:
        return 0

CLASSES = ('__background__', 'gate','redbuoy','greenbuoy','yellowbuoy','path','gateinv','torpedoboard','binbannana','binlightning','bincan','binorange',)#old
#CLASSES = ('__background__', 'gate','redbuoy','greenbuoy','yellowbuoy','path','gateinv','torpedoboard2016','torpedoboard2016cover','n','s','e','w','torpedoboard','binbannana','binlightning','bincan','binorange')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def vis_detections_video(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0: #is that class found?
        return im

    for i in inds: #Go through each of that class item
        bbox = dets[i, :4]
        score = dets[i, -1]

	print class_name
	print score
        objFound =0 
        for fObject in found_objects:
            if fObject.name ==class_name: #Object with that name already exists
                if abs(bbox[0] - fObject.px_location[0][0]) < 300: #
                    if abs(bbox[1] - fObject.px_location[0][1]) < 300:
                        fObject.confidence = score
                        fObject.increment_life()
                        fObject.inFrame = True
                        fObject.prev_px_location =   fObject.px_location
                        fObject.update_px_location(((bbox[0],bbox[1]),(bbox[2],bbox[3])))

                        fObject.average_location(bbox[0],bbox[1],bbox[2],bbox[3])
                       
                        fObject.width = abs(bbox[2]-bbox[0])
                        fObject.height = abs(bbox[3]-bbox[1])
                        fObject.area = fObject.width * fObject.height
                        fObject.objectCenter = (int(bbox[0]+fObject.width/2),int(bbox[1]+fObject.height/2))
                        fObject.time_not_seen = 0

                        #GIVES ME CANCER!!! SHOULD BE CLASS WITH LOOKUPS
                        if fObject.name=="gateinv":
                            objSize = 240.0 # CM Width of Object
                        elif fObject.name =="torpedoboard":
                            objSize = 120.0
                        elif fObject.name =="gate":
                            objSize = 305.0
                        elif fObject.name =="path":
                            objSize = 15.0
                        else: #buoys
                            objSize = 20.0
                        

                        fObject.shift_size(fObject.width)
              
                        fObject.update_distance(distance_to_camera(objSize,cameraProperties.FOCAL_LENGTH,fObject.average_size))
                        fx = 321.147192922
                        fy = 327.401895861
                        cx = 442.213579778
                        cy = 332.232842003
                        c_x =cx
                        c_y = cy
                        z_world = fObject.distance
                        f_x = fx
                        f_y = fy
                        x_screen,y_screen = fObject.objectCenter
                        x_world = (x_screen - c_x) * z_world / f_x
                        y_world = (y_screen - c_y) * z_world / f_y
                        #print x_world,y_world,z_world
                        fObject.rl_location=(x_world,y_world,z_world)
                        
                        objFound =1
                        break 

        if not objFound: #First time any object is found
            newObject = detectedObject(class_name)
            newObject.update_px_location(((bbox[0],bbox[1]),(bbox[2],bbox[3])))
            found_objects.append(newObject)

                        
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)

        cv2.rectangle(im,(int(bbox[0]),int(bbox[1]-20)),(int(bbox[0]+200),int(bbox[1])),(180,180,180),-1)
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]),int(bbox[1]-2)),cv2.FONT_HERSHEY_SIMPLEX,.75,(200,0,0))#,cv2.CV_AA)
    return im

def demo_video(net, im):
    """Detect object classes in an image using pre-computed object proposals."""


    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.0001
    NMS_THRESH = 0.2
    for fObject in found_objects:
        fObject.inFrame = False

    print 'new frame \n\n\n\n'
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]

        cls_scores = scores[:, cls_ind]
	#print cls_scores
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        im=vis_detections_video(im, cls, dets, thresh=CONF_THRESH)


    for fObject in found_objects:
        if not fObject.inFrame:
            fObject.time_not_seen +=1
           
    #cv2.circle(im,(1920/2,1200/2),int(5),(0,0,255),thickness=-1,lineType =8,shift=0) #center dot
    textColor = (0,121,255)   
    textColor = (200,102,38)
    #cv2.putText(im,'Num Objects: {:d}'.format(len(found_objects)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,.75,textColor)#,cv2.CV_AA
    count =3
    #for fObject in found_objects:
    #    #Prints info about all objects
    #    cv2.putText(im,'{:s} - L: {:d} - D: {:d}'.format(fObject.name, fObject.life, fObject.time_not_seen),(50,25*count),cv2.FONT_HERSHEY_SIMPLEX,.75,textColor)#,cv2.CV_AA
    #    count +=1
    #    if fObject.inFrame: #Prints info for only currently seen objects
    #        cv2.putText(im,'{:s} - Life: {:.0f} - Loc: ({:.0f},{:.0f},{:.0f})'.format(fObject.name, fObject.life,fObject.rl_location[0],fObject.rl_location[1],fObject.rl_location[2]),(int(fObject.px_location[0][0]),int(fObject.px_location[0][1])-50),cv2.FONT_HERSHEY_SIMPLEX,.75,textColor)
    #        cv2.putText(im,'Width: {:1f} - Avg Width: {:.1f}'.format(fObject.width, fObject.average_size),(int(fObject.px_location[0][0]),int(fObject.px_location[0][1])-75),cv2.FONT_HERSHEY_SIMPLEX,.75,textColor)
    #        cv2.putText(im,'Distance: {:1f} - Area: {:.1f}'.format(fObject.distance, fObject.area),(int(fObject.px_location[0][0]),int(fObject.px_location[0][1])-100),cv2.FONT_HERSHEY_SIMPLEX,.75,textColor)
     #       cv2.circle(im,fObject.objectCenter,int(5),(255,0,255),thickness=-1,lineType =8,shift=0)
            #cv2.circle(im,(int(fObject.prev_px_location[0][0]),int(fObject.prev_px_location[0][1])),int(5),(255,0,255),thickness=-1,lineType =8,shift=0)
      #      cv2.rectangle(im,(fObject.x1_average,fObject.y1_average),(fObject.x2_average,fObject.y2_average),(230,224,176),2)

    if PRINT_STATS:
        for fObject in found_objects:
            pass

    if RECORD:
        cv2.imwrite(os.path.join('output',str(time.time())+'.jpg'),im)
    
    if GUI:
        cv2.imshow('ret',im)
        cv2.waitKey(20)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


def getImage(camera,cameraIndex=0, format='bgr', scale=1.0, windowName='Live Video'):
    while cv2.waitKey(1) == -1:
        image = camera.GrabNumPyImage(format)
        if scale != 1.0:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            return image
        return image

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    cameraProperties = camera("Blackfly", 10, 100, 63, 4.0, 1)
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437520092_14466389'  #1
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437520291_14466389'  #2
    imagePath = '/home/goring/Documents/DataSets/Sub/2016/Transdec/1469722328_14466387'  #3
    imagePath = '/home/goring/Documents/DataSets/Sub/2016/Transdec/1469737334_14466387'  #4
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437581376_14466389'  #5
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437520549_14466389'  #6
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437520546_14466387'  #7
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437520291_14466389'  #8
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437520089_14466387'  #9
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437508031_14466389'  #10
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437491784_14466387'  #11
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437491781_14466389'  #12
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/buoys'  #12

    imageFiles = []
    for f in os.listdir(imagePath):
        if f.endswith('jpg') or f.endswith('.jpeg'):
            imageFiles.append(f)
    imageFiles = sorted(imageFiles)

    if GUI:
        cv2.namedWindow('ret',0)
    
    args = parse_args()
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    prototxt = '/home/goring/Documents/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/12/test.prototxt'
    caffemodel = '/home/goring/Documents/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/idk/50000robosub2015.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    if INPUT_MODE ==1:
        import pyfly2
        context = pyfly2.Context()
        if context.num_cameras < 1:
            raise ValueError('No cameras found')
        camera = context.get_camera(0)
        camera.Connect()
        camera.StartCapture()
        while True:
            image = getImage(camera)
            demo_video(net,image)
    elif INPUT_MODE ==0:
        for imageName in imageFiles:
            image = cv2.imread(os.path.join(imagePath,imageName))
            demo_video(net,image)
    elif INPUT_MODE ==2:
        try: ##Windows/OSX
            from PIL import ImageGrab
            while True:
                img = ImageGrab.grab()
                image = np.array(img)
                demo_video(net,image)
        except: #LINUX
            print "Does not work on Linux"
            
            ##import pyscreenshot as ImageGrab
            ##while True:
            ##    try:
            ##        im=ImageGrab.grab()
            ##        image = np.array(im)
            ##        cv2.imshow('fuck',image)
            ##        cv2.waitKey(20)
            ##        demo_video(net,image)
            ##    except:
            ##       pass
