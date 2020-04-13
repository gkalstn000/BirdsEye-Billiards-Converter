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
#import cv2

from PIL import Image


from object_detection.utils import ops as utils_ops

# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util


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

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


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



def img_cut(image_path) : 
    image = Image.open(image_path)
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