#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:52:14 2020

@author: gkalstn
"""


import cv2
import numpy as np
import numpy.linalg as lin

'''
img_path = '/Users/gkalstn/capstone/object_detection/special_res/1.jpg'
#img_path = '/Users/gkalstn/capstone/imgs/before_results.png'

image_np = cv2.imread(img_path)
'''







def cord_edge(image_np) :
    h, w = image_np[:,:,0].shape
    
    if (h > 1000) or (w > 1000) :
        h, w = int(h/3), int(w/3)
        image_np = cv2.resize(image_np, dsize=(w, h), interpolation=cv2.INTER_AREA)

    no_background = np.zeros((h, w))

    for i in range(h) :
        for j in range(w) :
            if (image_np[i,j,0] < image_np[i,j,1]) and (image_np[i,j,1] < image_np[i,j,2] and (image_np[i,j,0]<=44)) and (image_np[i,j,2]>170):
                no_background[i, j] = image_np[i, j, 0]

        
   

    for i in range(h) :
        for j in range(w) :
            if no_background[i, j] != 0 :
                no_background[i, j] = 200



    w_list = np.linspace(0, w-1, 17)
    h_list = np.linspace(0, h-1, 17)


    right = []
    left = []

#아래에서 검사 찾은 변이 2개 or 1개
    for i in h_list :
        i = int(i)
        lists = no_background[i, :].tolist()
    
        if(not (200 in lists)) :
            continue
    
        for j in range(w) :
            if no_background[i, w-1-j] == 200 :
                right.append([i, w-1-j])
                break

        for j in range(w) :
            if no_background[i, j] == 200 :
                left.append([i, j])
                break
        



    s1 = []
    s2 = []
    s3 = []
    s4 = []

    piv1 = test_line_num(right)
    piv2 = test_line_num(left)

    if piv1 == 0 :
        s1 = right
        if left[-1][-1] > left[0][-1] :
            s3 += left[:piv2]
            s2 += left[piv2+4:]
        else :
            s2 += left[:piv2]
            s3 += left[piv2+4:]
        
        for i in w_list :
            i = int(i)
            lists = no_background[i, :].tolist()
    
            if(not (200 in lists)) :
                continue
    
            for j in range(h) :
                if no_background[h-j-1, i] == 200 :
                    s4.append([h-j-1, i])
                    break
        
        
    
    else :
        s1 += right[:piv1]
        s2 += right[piv1+4:]
        s4 += left[:piv2]
        s3 += left[piv2+4:]
    
    

    p1 = equation(s1, s2)
    p2 = equation(s2, s3)
    p3 = equation(s3, s4)
    p4 = equation(s4, s1)


    
    point_list = [p1, p2, p3, p4]
    print(point_list)
    return point_list




def getcos(combination_list) :
    v1 = combination_list[1] - combination_list[0]
    v2 = combination_list[2] - combination_list[0]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1, ord=2)*np.linalg.norm(v2, ord=2))
    
    return abs(cos)

def test_line_num(point_lists) :
    for i in range(len(point_lists)-3) :
        if getcos(np.array(point_lists[i:i+3])) < 0.999 :
            print('detected 2 lines')
            return i
            
    
    print('detected 1 lines')
    return 0


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


import matplotlib.pyplot as plt

def graph_(lists) :
    x = []
    y = []
    for i in lists :
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,  # x
             y,  # y 
             linestyle='none', 
             marker='o', 
             markersize=10,
             color='blue', 
             alpha=0.5)
    plt.show()
    