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

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2,time
import argparse
global lastColor
lastColor = ''
colorOrder = ['']
CLASSES = ('__background__', 'lighttower','redtower','greentower','yellowtower','blacktower','bluetower')        
#CLASSES = ('__background__', 'diver','gate','redbuoy','greenbuoy','yellowbuoy','pathfront','gateinv')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections_video(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    global lastColor
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
	if 'red' in class_name or 'green' in class_name or 'yellow' in class_name or 'black' in class_name:
	    #print class_name
	    if class_name not in colorOrder[-1]:
	        colorOrder.append(class_name)
		print class_name
		lastColor = class_name
	cv2.putText(im,'Current Color',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
	txtColor = (0,0,0)	
	if 'red' in lastColor:
	    txtColor = (0,0,255)
        if 'green' in lastColor:
	    txtColor = (0,255,0)
        if 'yellow' in lastColor:
	    txtColor = (0,255,255)
	cv2.putText(im,'{:s}'.format(lastColor),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,txtColor)
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
    CONF_THRESH = 0.95

    NMS_THRESH = 0.05
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
    cv2.imshow('ret',im)
    cv2.waitKey(20)
    #if key ==ord('q'):
#	print '\n\n\n'
#        print colorOrder#
	#print '\n\n\n'
    cv2.imwrite(os.path.join('output2',str(time.time())+'.jpg'),im)

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

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    videoFile = cv2.VideoCapture('lighttower.avi')
   # cv2.namedWindow('ret',0)
    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
     #                       'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')

    prototxt = '/home/goring/Documents/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/lightTower2/test.prototxt'
    #print prototxt
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
    #                          NETS[args.demo_net][1])
    #print 'other caffe model'
    #print caffemodel
    #caffemodel = os.path.join(cfg.DATA_DIR,'faster_rcnn_end2end','voc_2007_trainval','vgg16_faster_rcnn_iter_10000.caffemodel')
    caffemodel = '/home/goring/Documents/py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/Tower1/vgg16_faster_rcnn_iter_2500.caffemodel'
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
    imageFiles = []
    imageDirectory = "/home/goring/Desktop/Footage/Vision/AVI/Good/videoframes/123"
    for f in os.listdir(imageDirectory):
        if f.endswith('jpg') or f.endswith('.jpeg'):
            imageFiles.append(f)
    imageFiles = sorted(imageFiles)
    for imageName in imageFiles:
        image = cv2.imread(os.path.join(imageDirectory,imageName))
        demo_video(net,image)
 #   plt.show()
