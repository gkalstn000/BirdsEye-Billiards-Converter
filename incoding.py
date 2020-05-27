#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:33:58 2020

@author: gkalstn
"""


import numpy as np
from distutils.version import StrictVersion

import cv2
import tensorflow.compat.v1 as tf
import time
import sys


import cordinate.edge_detect2 as ed
import cordinate.point_order as po
import object_detection.cut_obj as co
import trans.imgwarp2 as iw



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#This is needed since the code is stored in the object_detection    folder.
sys.path.append("..")
#from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def ball_cord(cord) :
    w1, h1, w2, h2 = cord
    
    p1 = np.array([h2, w2])
    p2 = np.array([h2, w1])
    
    return p1+p2/2

#Detection using tensorflow inside write_video function

def write_video():

    filename = './faster_rcnn_inception_400000step.mp4'
#    codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')

    codec = cv2.VideoWriter_fourcc(*'MJPG')
    cap = cv2.VideoCapture('./test_video/video5.mp4')

    framerate = round(cap.get(5),2)
    
    #w = int(cap.get(3))
    #h = int(cap.get(4))
    w, h = 348, 630
    resolution = (w, h)

    VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)    

    ################################
    # # Model preparation 

    # ## Variables
    # 
    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
    # 


    # What model to download.
    PATH_TO_FROZEN_GRAPH = './object_detection/fine_tuned_model/frozen_inference_graph.pb'
    MODEL_NAME = 'faster_rcnn_inception'
    print("loading model from " + MODEL_NAME)


    # ## Load a (frozen) Tensorflow model into memory.

    time_graph = time.time()
    print('loading graphs')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("tempo build graph = " + str(time.time() - time_graph))
    
            
    
    
    
    
    
    # ## Loading label map

    ikk = []
    ################################
    
    with tf.Session(graph=detection_graph) as sess:

        with detection_graph.as_default(): 

            while (cap.isOpened()):
                
                
                time_loop = time.time()
                print('processing frame number: ' + str(cap.get(1)))
                time_captureframe = time.time()
                ret, image_np = cap.read()
                
                
                print("time to capture video frame = " + str(time.time() - time_captureframe))
                if (ret != True):
                    break
                
                #image cut part=========================================
  #              output_dict = co.run_inference_for_single_image(image_np, detection_graph)
                        # Read frame from camera
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detections
                # Actual detection.
               
            
                (boxes, scores, classes) = sess.run(
                    [boxes, scores, classes],
                    feed_dict={image_tensor: image_np_expanded})
                output_dict = {'detection_boxes' : np.resize(boxes, (boxes.shape[1], boxes.shape[2])),
                               'detection_scores' : np.resize(scores, (scores.shape[1])),
                               'detection_classes' : np.resize(classes, classes.shape[1])}    
                
                
                
                
                cordinates = co.class_cordinate(output_dict)
                img_height, img_width, img_channel = image_np.shape
                absolute_coord = []
                
                for i in cordinates.keys() :
                    ymin, xmin, ymax, xmax = cordinates[i]
                    x_up = int(xmin*img_width)
                    y_up = int(ymin*img_height)
                    x_down = int(xmax*img_width)
                    y_down = int(ymax*img_height)
                    absolute_coord.append([x_up,y_up,x_down,y_down])
                
                bounding_box_img = []
                c = absolute_coord[0]
                bounding_box_img = image_np[c[1]:c[3], c[0]:c[2],:]
    
                
                #[h, w]
                red = [absolute_coord[1][3] , (absolute_coord[1][0]+absolute_coord[1][2])//2 ]
                white = [absolute_coord[2][3] , (absolute_coord[2][0]+absolute_coord[2][2])//2 ]
                yellow = [absolute_coord[3][3] , (absolute_coord[3][0]+absolute_coord[3][2])//2 ]
        
                points = [white, red, yellow]
    
                result_dict = {'image_np' : bounding_box_img,
                               'points' : points}
                #======================================================
                
                image_np = result_dict['image_np']
                points = result_dict['points']
#                if i == 0 :
                #cordinate part=======================================
                if cap.get(1) == 1.0 :
                    try :
                        result = np.array(ed.cord_edge(image_np))
                        result = po.point_order(result)
                    except Exception :
                        continue
                #=====================================================
                
                
                    ikk += [(result[0][0] + c[1]/3, result[0][1] + c[0]/3),
                            (result[2][0] + c[1]/3, result[2][1] + c[0]/3),
                            (result[1][0] + c[1]/3, result[1][1] + c[0]/3),
                            (result[3][0] + c[1]/3, result[3][1] + c[0]/3)]
                    print('ikk : ', ikk)
                
                
                    
#                print('all_points : ', all_points)
                

                balls = [(points[0][0]/3, points[0][1]/3),
                         (points[1][0]/3, points[1][1]/3),
                         (points[2][0]/3, points[2][1]/3)]

                
                
                
  #              print('edge points : ', ikk)
   #             print('balls : ', balls)
  #              print('input : ', ikk+balls)
                
 #               print('ball detect num : ', len(balls))
                try :
                    final_image = iw.warp(ikk+balls)
                except Exception :
                    continue
                
                cv2.imwrite('./vimages/img'+str(int(cap.get(1)))+'.jpg', final_image)
              
              #  print('final_image shape : ', final_image.shape)
                time_writeframe = time.time()
                VideoFileOutput.write(final_image)
                print("time to write a frame in video file = " + str(time.time() - time_writeframe))

                print("total time in the loop = " + str(time.time() - time_loop))
                

    cap.release()
    VideoFileOutput.release()
    print('done')
    
write_video()
