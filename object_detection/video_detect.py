#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:14:58 2020

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


from utils import label_map_util

from utils import visualization_utils as vis_util

#Detection using tensorflow inside write_video function

def write_video():

    filename = 'output/faster_rcnn_inception_400000step_2.avi'
    codec = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
    cap = cv2.VideoCapture('./test_video/video2.mp4')
    framerate = round(cap.get(5),2)
    w = int(cap.get(3))
    h = int(cap.get(4))
    resolution = (w, h)

    VideoFileOutput = cv2.VideoWriter(filename, codec, framerate, resolution)    

    ################################
    # # Model preparation 

    # ## Variables
    # 
    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
    # 


    # What model to download.
    PATH_TO_FROZEN_GRAPH = './fine_tuned_model/frozen_inference_graph.pb'
    PATH_TO_LABEL_MAP = './label_map.pbtxt'
    NUM_CLASSES = 4
    MODEL_NAME = 'faster_rcnn_resnet101'
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
              image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
              boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
              scores = detection_graph.get_tensor_by_name('detection_scores:0')
              classes = detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = detection_graph.get_tensor_by_name('num_detections:0')
              # Actual detection.
              time_prediction = time.time()
              (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
              print("time to predict = " + str(time.time() - time_prediction))
              # Visualization of the results of a detection.
              time_visualizeboxes = time.time()
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
              print("time to generate boxes in a frame = " + str(time.time() - time_visualizeboxes))


              time_writeframe = time.time()
              VideoFileOutput.write(image_np)
              print("time to write a frame in video file = " + str(time.time() - time_writeframe))

              print("total time in the loop = " + str(time.time() - time_loop))

    cap.release()
    VideoFileOutput.release()
    print('done')
    
write_video()
  
    
