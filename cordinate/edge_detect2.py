#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:14:58 2020

@author: gkalstn
"""


import cv2
import numpy as np
import numpy.linalg as lin



def getcos(combination_list) :
    v1 = combination_list[1] - combination_list[0]
    v2 = combination_list[2] - combination_list[0]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1, ord=2)*np.linalg.norm(v2, ord=2))
    
    return abs(cos)

def test_line_num(point_lists) :
    for i in range(len(point_lists)-3) :
        if getcos(np.array(point_lists[i:i+3])) < 0.999 :
            print('2 lines detected')
            return i
            
    
    print('1 lines detected')
    
    return 0


def equation(point_list1, point_list2) :
    try :
        x11 = point_list1[0][0]
        y11 = point_list1[0][1]
        x12 = point_list1[1][0]
        y12 = point_list1[1][1]
    
        x21 = point_list2[0][0]
        y21 = point_list2[0][1]
        x22 = point_list2[1][0]
        y22 = point_list2[1][1]
    except IndexError :
        print('pl1 : ', point_list1)
        print('pl2 : ', point_list2)
        return None
    
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
        y.append(-i[0])
        
    
    plt.plot(x,  # x
             y,  # y 
             linestyle='none', 
             marker='o', 
             markersize=10,
             color='blue', 
             alpha=0.5)
    

    
        
    
    plt.savefig('./graph/savefig_default.png')
    plt.close()
    
'''    
img_path = '/Users/gkalstn/capstone/test_images/img0.jpeg'

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
            if (image_np[i,j,0] > image_np[i,j,1]) and (image_np[i,j,1] > image_np[i,j,2] and (image_np[i,j,0] -image_np[i, j, 1] > 70)) and (image_np[i,j,2]<170):
                no_background[i, j] = image_np[i, j, 0]

        
   

    for i in range(h) :
        for j in range(w) :
            if no_background[i, j] != 0 :
                no_background[i, j] = 200



    w_list = np.linspace(0, w-1, 30)
    h_list = np.linspace(0, h-1, 30)


    bot = []
    top = []

#아래에서 검사 찾은 변이 2개 or 1개
    for i in w_list :
        i = int(i)
        lists = no_background[:, i].tolist()
    
        if(not (200 in lists)) :
            continue
    
        for j in range(h) :
            if no_background[h-1-j, i] == 200 :
                bot.append([h-1-j, i])
                break

        for j in range(h) :
            if no_background[j, i] == 200 :
                top.append([j, i])
                break
        



    s1 = []
    s2 = []
    s3 = []
    s4 = []


    print('bot detect ', end='')
    piv1 = test_line_num(bot)
    print('top detect ', end='')
    piv2 = test_line_num(top)
    


    if piv1 == 0 :
        s1 = bot
        if top[0][0] < top[-1][0] : # 윗변의 왼쪽이 더 높은것
            s3 += top[:piv2]
            s2 += top[piv2+4:]
        
            for i in h_list :
                i = int(i)
                lists = no_background[i, :].tolist()
                
                if(not (200 in lists)) :
                    continue
    
                for j in range(w) :
                    if no_background[i, j] == 200 :
                        s4.append([i, j])
                        break
        else :
            s2 += top[:piv2]
            s3 += top[piv2+4:]
            
            for i in h_list :
                i = int(i)
                lists = no_background[i, :].tolist()
    
                if(not (200 in lists)) :
                    continue
    
                for j in range(w) :
                    if no_background[i, w-1-j] == 200 :
                        s4.append([i, w-1-j])
                        break        

        
        
    
    else :
        s1 += bot[:piv1]
        s2 += bot[piv1+4:]
        s4 += top[:piv2]
        s3 += top[piv2+4:]
    '''
    print('s1 : ', s1)
    print('s2 : ', s2)
    print('s3 : ', s3)
    print('s4 : ', s4)
    
    print('s1 : ', graph_(s1))
    print('s2 : ', graph_(s2))
    print('s3 : ', graph_(s3))
    print('s4 : ', graph_(s4))
    '''
    p1 = equation(s1, s2)
    p2 = equation(s2, s3)
    p3 = equation(s3, s4)
    p4 = equation(s4, s1)


    
    point_list = [p1, p2, p3, p4]
    return point_list


