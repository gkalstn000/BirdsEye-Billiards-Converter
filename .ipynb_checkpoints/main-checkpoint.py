#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:10:59 2020

@author: gkalstn
"""
#git testingimport time
 # 시작 시간 저장


import numpy as np
import cordinate.Billiards_Detect_test as bd
import cordinate.point_order as po
#import sys
#import os
#import tensorflow.compat.v1 as tf
import cv2
from PIL import Image
import object_detection.cut_obj as co
import trans.imgwarp2 as iw
import trans.show_result as sr


import time
start = time.time() 
image_path = './test_images/img2.jpeg'
result_dict = co.img_cut(image_path)

image_np = result_dict['image_np']
points = result_dict['points']



#import math
#import matplotlib.pyplot as plt

result = np.array(bd.Detecting(image_np))
result = po.point_order(result)

all_points = [(result[0][0], result[0][1]),
             (result[1][0], result[1][1]),
             (result[2][0], result[2][1]),
             (result[3][0], result[3][1])]

all_points.append((points[0][0], points[0][1]))
all_points.append((points[1][0], points[1][1]))
all_points.append((points[2][0], points[2][1]))
print('당구대', all_points[:4])
print('당구공', all_points[4:])
'''
<<<<<<< HEAD
final_image = iw.warp(all_points)
img = Image.fromarray(final_image, 'RGB') 
img.save('./test_image_result/result.jpeg')
=======
'''
final_image = iw.warp(all_points)
# cv2.imshow('test',final_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
#img = Image.fromarray(final_image, 'RGB')
#img.save('./test_image_result/results.jpeg')


#image_path = './test_images/img6.jpeg'
final_image = iw.warp(all_points)
print('image.shape   =====',final_image.shape)
cv2.imshow('w', final_image)

#sr.show_result(image_path)
print("cost time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
#>>>>>>> 9a9683fe9d55e0cab75d9b16c39e9c76132d7eda
'''
