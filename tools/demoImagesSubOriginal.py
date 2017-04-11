#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
USE_CAMERA=0
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

CLASSES = ('__background__', 'gate','redbuoy','greenbuoy','yellowbuoy','path','gateinv','torpedoboard','binbannana','binlightning','bincan','binorange',)#old

#CLASSES = ('__background__', 'gate','redbuoy','greenbuoy','yellowbuoy','path','gateinv','torpedoboard2016','torpedoboard2016cover','n','s','e','w','torpedoboard','binbannana','binlightning','bincan','binorange')

#CLASSES = ('__background__','redbuoy')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections_video(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #print 'ahahahah'
        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
        cv2.rectangle(im,(int(bbox[0]),int(bbox[1]-20)),(int(bbox[0]+200),int(bbox[1])),(10,10,10),-1)
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]),int(bbox[1]-2)),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255))#,cv2.CV_AA)
    return im

def demo_video(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #ret, im = videoFile.read()
    #cv2.imshow('bla',im)
    #cv2.waitKey(20)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.80

    NMS_THRESH = 0.2
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
    #cv2.imwrite(os.path.join('tools/output',str(time.time())+'.jpg'),im)
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
    #videoFile = cv2.VideoCapture('lighttower.avi')
    #imagePath = '/home/goring/Documents/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages'
    #imagePath = '/home/goring/Documents/DataSets/Sub/2015/ERAUPool/1436740139_14466389'
    imagePath = '/home/goring/Documents/DataSets/Sub/2015/Transdec/1437520092_14466389'
    imageFiles = []
    for f in os.listdir(imagePath):
        if f.endswith('jpg') or f.endswith('.jpeg'):
            imageFiles.append(f)
    imageFiles = sorted(imageFiles)
    args = parse_args()
    cv2.namedWindow('ret',0)
    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
     #                       'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')

    prototxt = '/home/goring/Documents/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/12/test.prototxt'
    #print prototxt
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])
    #print 'other caffe model'
    #print caffemodel
    #caffemodel = os.path.join(cfg.DATA_DIR,'faster_rcnn_end2end','voc_2007_trainval','vgg16_faster_rcnn_iter_50000.caffemodel')
#    caffemodel = '/home/goring/Documents/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_10000.caffemodel'
    caffemodel = '/home/goring/Documents/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/50000robosub2015.caffemodel'
    #print caffemodel
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    #im_names = ['red.jpg','green.jpg','yellow.jpg','black.jpg']
    #im_names = ['000012.jpg','000013.jpg','000140.jpg','000141.jpg']
    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    #for im_name in im_names:
    #    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #    print 'Demo for data/demo/{}'.format(im_name)
        #demo(net, im_name)
    if USE_CAMERA:
        import pyfly2
	context = pyfly2.Context()
	if context.num_cameras < 1:
	    raise ValueError('No cameras found')
        camera = context.get_camera(0)
	camera.Connect()
	camera.StartCapture()
	while(1):
	    image = getImage(camera)
            demo_video(net,image)
    else:
        for imageName in imageFiles:
	    image = cv2.imread(os.path.join(imagePath,imageName))
            demo_video(net,image)
 #   plt.show()