#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:19:23 2020

@author: gkalstn
"""

import numpy as np
from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
from distutils.version import LooseVersion, StrictVersion
import cv2
from imutils import paths
import tensorflow.compat.v1 as tf
import time
import re
import sys

#This is needed since the code is stored in the object_detection    folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

#Detection using tensorflow inside write_video function


import sys
import os
#import object_detection.cut_obj as co
import cordinate.Billiards_Detect_test as bd
import cordinate.point_order as po
import trans.imgwarp2 as iw
import math
import matplotlib.pyplot as plt


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                    ]:
              
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
          
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
        
        
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

              # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

              # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def class_cordinate(output_dict) :
    detection_boxes = output_dict['detection_boxes']
    detection_scores = output_dict['detection_scores']
    detection_classes = output_dict['detection_classes']

    class_1_max_score = 0
    class_2_max_score = 0
    class_3_max_score = 0
    class_4_max_score = 0
    
    class_1_ind = 0
    class_2_ind = 0
    class_3_ind = 0
    class_4_ind = 0


    for i in range(len(detection_classes)) :
        if detection_classes[i] == 1 :
            if class_1_max_score < detection_scores[i]:
                class_1_max_score = detection_scores[i]
                class_1_ind = i
        elif detection_classes[i] == 2 :
            if class_2_max_score < detection_scores[i]:
                class_2_max_score = detection_scores[i]
                class_2_ind = i
        elif detection_classes[i] == 3 :
            if class_3_max_score < detection_scores[i]:
                class_3_max_score = detection_scores[i]
                class_3_ind = i
        else :
            if class_4_max_score < detection_scores[i]:
                class_4_max_score = detection_scores[i]
                class_4_ind = i

    result = {'table' : detection_boxes[class_2_ind],
             'red_ball' : detection_boxes[class_1_ind],
             'white_ball' : detection_boxes[class_3_ind],
             'yellow_ball' : detection_boxes[class_4_ind]}
    return result


filename = './test_video_result/video_result.avi'
codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
cap = cv2.VideoCapture('./object_detection/test_video/video2.mp4')
framerate = round(cap.get(5),2)
w, h = 315, 612
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
PATH_TO_LABEL_MAP = './object_detection/label_map.pbtxt'
NUM_CLASSES = 4
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

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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
                
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            #image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            #=========================
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            cordinates = class_cordinate(output_dict)
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
            red = [(absolute_coord[1][0]+absolute_coord[1][2])//2 - c[0], (absolute_coord[1][1]+absolute_coord[1][3])//2 - c[1]]
            white = [(absolute_coord[2][0]+absolute_coord[2][2])//2 - c[0], (absolute_coord[2][1]+absolute_coord[2][3])//2 - c[1]]
            yellow = [(absolute_coord[3][0]+absolute_coord[3][2])//2 - c[0], (absolute_coord[3][1]+absolute_coord[3][3])//2 - c[1]]
    
            points = [white, red, yellow]
#            image_np = bounding_box_img
    

            #=========================
            
            result = np.array(bd.Detecting(bounding_box_img))
            result = po.point_order(result)
            
            all_points = [(result[0][0], result[0][1]),
                         (result[1][0], result[1][1]),
                         (result[2][0], result[2][1]),
                         (result[3][0], result[3][1])]

            all_points.append((points[0][0], points[0][1]))
            all_points.append((points[1][0], points[1][1]))
            all_points.append((points[2][0], points[2][1]))
            
            ani = iw.warp(all_points)
                


            time_writeframe = time.time()
            VideoFileOutput.write(ani)
            print("time to write a frame in video file = " + str(time.time() - time_writeframe))

            print("total time in the loop = " + str(time.time() - time_loop))

cap.release()
VideoFileOutput.release()
print('done')


    