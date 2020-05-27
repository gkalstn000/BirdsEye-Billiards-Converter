#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 00:14:25 2020

@author: gkalstn
"""

import numpy as np
#import sys
#import os
import tensorflow.compat.v1 as tf
from PIL import Image
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
#from object_detection.utils import visualization_utils as vis_util
'''

PATH_TO_FROZEN_GRAPH = './object_detection/fine_tuned_model/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = './object_detection/label_map.pbtxt'
NUM_CLASSES = 4
MODEL_NAME = 'faster_rnn_inception'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
        
label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
'''
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, detection_graph):
    with detection_graph.as_default():
        with tf.Session() as sess:
            # Read frame from camera
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
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



def img_cut(image_path, detection_graph) : 
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    image = image.resize((845,526))
    image_np = load_image_into_numpy_array(image)
    
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
    
    result_dict = {'image_np' : bounding_box_img,
                   'points' : points}
    
    return result_dict
'''
import cv2

PATH_TO_FROZEN_GRAPH = './fine_tuned_model/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = './label_map.pbtxt'
NUM_CLASSES = 4
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
        
label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
image_path = '/Users/gkalstn/capstone/object_detection/special_case/img3.jpg'
dic = img_cut(image_path)
img = dic['image_np']
cv2.imwrite('/Users/gkalstn/capstone/object_detection/special_res/img3.jpg', img)
'''

