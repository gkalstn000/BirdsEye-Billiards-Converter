#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:08:48 2020

@author: gkalstn
"""
import numpy as np
import math


def point_order(point_list) :    
    p1, p2, p3, p4 = point_list
    
    #p1을 기준으로
    A1 = p2 - p1
    A2 = p3 - p1
    A3 = p4 - p1
    
    #length of vector
    len_A1 = get_distance(p2, p1)
    len_A2 = get_distance(p3, p1)
    len_A3 = get_distance(p4, p1)
    

    
    angle_dict = {get_angle(A1, A2, len_A1, len_A2) : 'angle_A1A2',
                  get_angle(A1, A3, len_A1, len_A3) : 'angle_A1A3',
                  get_angle(A2, A3, len_A2, len_A3) : 'angle_A2A3'}
    
    
    tmp = list(angle_dict.keys())    
    biggest_angle = angle_dict.get(max(tmp))  
    key_point = 0
    
    #p1의 마주보는 각 구하기
    if biggest_angle == 'angle_A1A2':
        key_point = p4
    elif biggest_angle == 'angle_A1A3' :
        key_point = p3
    else :
        key_point = p2
    
    point_list = point_list.tolist()    
    point_list.remove(key_point.tolist())
    
    #p1과 key_point는 마주보는 각
    point_list = np.array(point_list)
    
    point_angle = {get_point_angle(point_list[1], p1, point_list[2]) : 'point1',
                   get_point_angle(point_list[1], key_point, point_list[2]) : 'point2',
                   get_point_angle(p1, point_list[1], key_point) : 'point3',
                   get_point_angle(p1, point_list[2], key_point) : 'point4'
                   }
        
    tmp = list(point_angle.keys())
    biggest_angle = angle_dict.get(max(tmp))
    
    if biggest_angle == 'point1' or 'point2' :
        #point3 or point4 중 긴다이 짧은다이 찾아서 리턴
        if biggest_angle == 'point1' :
            p = p1
            p_ = key_point
        else :
            p = key_point
            p_ = p1
            
        to_p3 = get_distance(point_list[1], p)
        to_p4 = get_distance(point_list[2], p)
        
        if to_p3 > to_p4 :
            #return np.array([p, point_list[2], point_list[1], p_])
            #return np.array([p, point_list[1], point_list[2], p_]) 이게 위아래 반대
            return np.array([point_list[2], p_, p, point_list[1]])
        else :
            #return np.array([p, point_list[1], point_list[2], p_])
            #return np.array([p, point_list[2], point_list[1], p_])
            return np.array([point_list[1], p_, p, point_list[2]])
    
        
    elif biggest_angle == 'point3' or 'point4' :
        #point1 or point2 중 긴다이 짧은다이 찾아서 리턴
        if biggest_angle == 'point3' :
            p = point_list[1]
            p_ = point_list[2]
        else :
            p = point_list[2]
            p_ = point_list[1]
        
        to_p1 = get_distance(p1, p)
        to_p2 = get_distance(key_point, p)
        
        if to_p1 > to_p2 :
            #return np.array([p, key_point, p1, p_])
            #return np.array([p, p1, key_point, p_])
            return np.array([key_point, p_, p, p1])
        else :
            #return np.array([p, p1, key_point, p_])
            #return np.array([p, key_point, p1, p_])
            return np.array([p1, p_, p, key_point])
        
    

#세 점중 가운데 점각 구하기
def get_point_angle(p1, p2, p3) :
    d1 = get_distance(p2, p1)
    d2 = get_distance(p2, p3)
    
    return get_angle(p1-p2, p3-p2, d1, d2)

#두 점사이 거리
#벡터의 길이
def get_distance(point1, point2) :
    return math.sqrt(math.pow(point1[0]-point2[0], 2) + math.pow(point1[1] -point2[1], 2))


#두 벡터사이 각 구하는 메서드
def get_angle(A, B, len_A, len_B) :
    angle = np.arccos(np.dot(A, B) / (len_A*len_B))
        
    return angle

def get_index(list, value) :
    k = 0
    for i in list :
        if i == value :
            return k
        else :
            k += 1
    if i == len(value) :
        print('그런 포인트 없음')


