#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:25:03 2020

@author: gkalstn
"""

import numpy as np
import sys
import os
import tensorflow.compat.v1 as tf
import cv2

from PIL import Image


from object_detection.utils import ops as utils_ops

# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


PATH_TO_FROZEN_GRAPH = '/Users/gkalstn/capstone/object_detection/fine_tuned_model/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = '/Users/gkalstn/capstone/object_detection/label_map.pbtxt'
NUM_CLASSES = 4
MODEL_NAME = 'ssd_mobilenet_v2'

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



def single_image_cut(image_path) :
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=1)  
    img_height, img_width, img_channel = image_np.shape
    absolute_coord = []
    THRESHOLD = 0.9 # adjust your threshold here
    N = len(output_dict['detection_boxes'])
    for j in range(N):
        if output_dict['detection_scores'][j] < THRESHOLD:
            continue
        box = output_dict['detection_boxes'][j]
        ymin, xmin, ymax, xmax = box
        x_up = int(xmin*img_width)
        y_up = int(ymin*img_height)
        x_down = int(xmax*img_width)
        y_down = int(ymax*img_height)
        absolute_coord.append((x_up,y_up,x_down,y_down))

    bounding_box_img = []
    for c in absolute_coord:
        bounding_box_img.append(image_np[c[1]:c[3], c[0]:c[2],:])
    tmp = []
    for i in range(len(bounding_box_img)) :
        tmp.append(len(bounding_box_img[i]))
    ind = tmp.index(max(tmp))
    
    cvt_img = cv2.cvtColor(bounding_box_img[ind], cv2.COLOR_BGR2RGB)
    
    return cvt_img