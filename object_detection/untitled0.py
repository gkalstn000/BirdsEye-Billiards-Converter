#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:52:14 2020

@author: gkalstn
"""

import cv2
import numpy as np
import numpy.linalg as lin


img_path = '/Users/gkalstn/capstone/object_detection/special_res/img2.jpg'
#img_path = '/Users/gkalstn/capstone/imgs/before_results.png'

origin = cv2.imread(img_path)
h, w = origin[:,:,0].shape
if (h > 1000) or (w > 1000) :
    h, w = int(h/3), int(w/3)
origin = cv2.resize(origin, dsize=(w, h), interpolation=cv2.INTER_AREA)


no_background = np.zeros((h, w, 3))

for i in range(h) :
    for j in range(w) :
        if (origin[i,j,0] > origin[i,j,1]) and (origin[i,j,1] > origin[i,j,2] and (origin[i,j,0]>190)) and (origin[i,j,1]<170):
            no_background[i, j] = origin[i, j]
#            print(origin[i,j])


        
            

wha = no_background[:,:,0]

for i in range(h) :
    for j in range(w) :
        if wha[i, j] != 0 :
            wha[i, j] = 200

piv = max(w, h)

a = wha[200, :].tolist()
a
a.index(200)

piv_x_list = np.linspace(0, piv-1, 17)

side1 = []
side2 = []



for i in piv_x_list : #i 가 행
    i = int(i)
    if piv == w :
        lists = wha[:, i].tolist()
        if (not (200 in lists)) or (lists.count(200) == 1) :
            continue
        
        for j in range(h) :
            if wha[j, i] == 200 :
                side1.append([j, i])
                break
            
        for j in range(h) :    
            if wha[h-1-j, i] == 200 :
                side2.append([h-1-j, i])
                break
        
        
        
    else :
        lists = wha[i, :].tolist()
        if (not (200 in lists)) or (lists.count(200) == 1):
            continue
        
        for j in range(w) :
            if wha[i, j] == 200 :
                side1.append([i, j])
                break
        for j in range(w) :
            if wha[i, w-1-j] == 200 :
                side2.append([i, w-1-j])
                break



s1 = []
s2 = []
s3 = []
s4 = []

def getcos(combination_list) :
    v1 = combination_list[1] - combination_list[0]
    v2 = combination_list[2] - combination_list[0]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1, ord=2)*np.linalg.norm(v2, ord=2))
    
    return abs(cos)


for i in range(len(side1)-3) :
    if getcos(np.array(side1[i:i+3])) < 0.999 :
        s1 += side1[:i]
        s2 += side1[i+4:]
        break

for i in range(len(side1)-3) :
    if getcos(np.array(side1[i:i+3])) < 0.999 :
        s4 += side2[:i]
        s3 += side2[i+4:]
        break
    

def equation(point_list1, point_list2) :
    x11 = point_list1[0][0]
    y11 = point_list1[0][1]
    x12 = point_list1[1][0]
    y12 = point_list1[1][1]
    
    x21 = point_list2[0][0]
    y21 = point_list2[0][1]
    x22 = point_list2[1][0]
    y22 = point_list2[1][1]
    
    A = np.array([[y12-y11, -x12+x11],
                  [y22-y21, -x22+x21]])    
    inv_A = lin.inv(A)
    X = np.array([x11*y12-x12*y11, x21*y22-x22*y21])
    
    result = np.dot(inv_A, X.T)
    
    return result


    
p1 = equation(s1, s2)
p2 = equation(s2, s3)
p3 = equation(s3, s4)
p4 = equation(s4, s1)





point_list = [p1, p2, p3, p4]



