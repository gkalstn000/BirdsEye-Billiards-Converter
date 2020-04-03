#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:10:59 2020

@author: gkalstn
"""

import numpy as np
import sys
import os
import tensorflow.compat.v1 as tf
import cv2
from PIL import Image

import object_detection.obj_cut as oc

image_path = './test_images/img137.jpg'



import time
start = time.time()  # 시작 시간 저장
image_np = oc.single_image_cut(image_path)
print("하민수 time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간


MODEL_NAME = 'ssd_mobilenet_v2'
PATH_TO_TEST_IMAGES_DIR = './test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, img) for img in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
TEST_SAVE_PATH = 'test_result/{}'.format(MODEL_NAME)


import cordinate.Billiards_Detect_test as bd
import math
import matplotlib.pyplot as plt


import time
start = time.time()  # 시작 시간 저장
result = bd.Detecting(image_np)
print("김건기 time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

import trans.imgwarp2 as iw


result.append([115, 169])
result.append([242, 149])
result.append([208, 117])


iw.img_warp(result, image_np)