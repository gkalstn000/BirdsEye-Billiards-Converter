#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:52:14 2020

@author: gkalstn
"""

import cv2
import numpy as np
import numpy.linalg as lin
import random


img_path = '/Users/gkalstn/capstone/object_detection/special_case/img19.jpg'
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















'''
cv2.imwrite('/Users/gkalstn/capstone/object_detection/special_res/img22.jpg', no_background)

cv2.imshow('origin', origin)
#cv2.imshow('after', after)
cv2.imshow('no_background', no_background)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

def getcos(combination_list) :
    v1 = combination_list[1] - combination_list[0]
    v2 = combination_list[2] - combination_list[0]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1, ord=2)*np.linalg.norm(v2, ord=2))
    
    return abs(cos)

def rand_list(w, n) :
    tmp = []
    ran_num = (random.randint(0, w))
    for i in range(n) :
        while ran_num in tmp :
            ran_num = random.randint(0, w)
        tmp.append(ran_num)
    return tmp


#image_path = '/Users/gkalstn/capstone/object_detection/special_res/img19t.png'
image_path = '/Users/gkalstn/no_background.png'
image_np = cv2.imread(image_path)

image_np.shape
h, w = image_np[:,:,0].shape

w_list = rand_list(w, 8)
h_list = rand_list(h, 8)


# 여기 중복 제거나중에 하기
combination_list = []

for i in w_list : #up to down
    for j in range(h) : 
        if (image_np[j, i, 0] > 180) and not([j, i] in combination_list):
            combination_list.append([j,i])
 #           print(j, i)
            break

for i in w_list : #down to up
    for j in range(h) : 
        if (image_np[h-1-j, i, 0] > 180) and not([h-1-j, i] in combination_list) :
            combination_list.append([h-1-j,i])
#            print(h-1-j, i)
            break


for i in h_list : #left to right
    for j in range(w) : 
        if (image_np[i, j, 0] > 180) and not([i, j] in combination_list) :
            combination_list.append([i,j])
 #           print(i, j)
            break

for i in h_list : #right to left
    for j in range(w) : 
        if (image_np[i, w-1-j, 0] > 180) and not([i, w-1-j] in combination_list) :
            combination_list.append([i,w-1-j])
#            print(i, w-1-j)
            break



from itertools import combinations

#포인트 조합에서 코사인 검사 후 중복 제거

point_list = []

#N, L
point_combinations1 = list(combinations(combination_list[:4]+combination_list[8:12],3))
len(point_combinations1)
import copy
tmp = copy.deepcopy(point_combinations1)
for i in point_combinations1 :
    if getcos(np.array(i)) < 0.9999 :
        tmp.remove(i)
point_list += tmp
        
#N, R
point_combinations2 = list(combinations(combination_list[:4]+combination_list[12:],3))
len(point_combinations2)
import copy
tmp = copy.deepcopy(point_combinations2)
for i in point_combinations2 :
    if getcos(np.array(i)) < 0.9999 :
        tmp.remove(i)
point_list += tmp
        
        
#S, L
point_combinations3 = list(combinations(combination_list[4:8]+combination_list[8:12],3))
len(point_combinations3)
import copy
tmp = copy.deepcopy(point_combinations3)
for i in point_combinations3 :
    if getcos(np.array(i)) < 0.999 :
        tmp.remove(i)
point_list += tmp        
        
#S, R
point_combinations4 = list(combinations(combination_list[:4]+combination_list[12:],3))
len(point_combinations4)
import copy
tmp = copy.deepcopy(point_combinations4)
for i in point_combinations4 :
    if getcos(np.array(i)) < 0.9999 :
        tmp.remove(i)
point_list += tmp        

ww = image_np[:,:,0]

no_same_points = []
no_same_points.append(point_list[0])

def check(list1, list2) :
    if (list1[0] in list2) or (list1[1] in list2) or (list1[2] in list2):
        return False
    else :
        return True
    
i = 1
piv = 0
while(True) :
    if i == len(point_list) -1 :
        break
    if check(point_list[piv], point_list[i]) :
        piv = i
        no_same_points.append(point_list[i])
    i += 1
    
    
    




'''
vector_list = []
for i in point_list :
    vector_list.append(round((i[1][1]-i[0][1])/(i[1][0]-i[0][0]), 1))
        

tmp = []
unoverlab = list(set(vector_list))

for i in unoverlab :
    tmp.append(vector_list.index(i))
    
result_point_list = []
for i in tmp :
    result_point_list.append(point_list[i])
'''

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

equation(result_point_list[1], result_point_list[2])




for i in result_point_list :
    print(i)


for i in point_list:
    print(getcos(np.array(i)))
    


vector_combinations = tmp

vectors = []
for i in vector_combinations :
    x = i[1][0]-i[0][0]
    y = i[1][1]-i[0][1]
    vectors.append(round(y/x, 2))

vectors
a = list(set(vectors))
