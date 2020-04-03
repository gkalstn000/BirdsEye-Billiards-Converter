#import
import sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
print(sys.version)
green_color = (0,255,0)
red_color = (0,0,255)
white_color = (255,255,255)
yellow_color = (0,255,255)
blue_color = (255,0,0)

# function define
def image_center(img_color):
    img_height, img_width = img_color.shape[:2]
    img_width = img_width/2
    img_height = img_height/2
    
    return int(img_width), int(img_height)
    
def color_range(hsv,per) :
    hsv_min = hsv-per
    hsv_max = hsv+per
    for i in range(hsv.shape[0]) :
        if hsv[i]+per > 255:
            hsv_max[i] = 255
        elif  hsv[i]-per < 0 :
            hsv_min[i] = 0
    return hsv_min,hsv_max
#좌표 디텍팅 문제시 이부분
def lines_points(img, lines, color=[255,0,0], thickness =3) :
    flag = 0
    for line in lines :
        for x1,y1,x2,y2 in line :
            if flag == 0 :
                a = [[x1,y1]]
                b = [[x2,y2]]
                result = np.append(a,b,axis=0)
                flag = 1
            else :
                a = [[x1,y1]]
                b = [[x2,y2]]
                a = np.append(a,b,axis=0)
                result=np.append(result,a,axis=0)
            
            cv2.line(img, (x1,y1),(x2,y2), color,thickness)
    print(result)
    return result
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap) :
    lines = cv2.HoughLinesP(img, rho,theta, threshold, np.array([]), 
                           minLineLength = min_line_len,
                           maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    result = lines_points(line_img, lines)
    
    return line_img, result

def max_min_point(array):
    shape=array.shape
    print(shape[0])
# 혹시 좌표 디택팅 오류면 이부분일 가능성 높다.

def distance(array,value): # 두점간의 거리를 구하는 function
    x1,y1 = array[0], array[1]
    x2,y2 = value[0], value[1]
    print(x1,y1,x2,y2)
    result =  math.sqrt(
        math.pow((x2 - x1), 2) + math.pow(y2 - y1,2))
    
    return result
def nearpos(array,value,flag):
    result = []
    lines = array.shape[0]
    if flag == 0 : # width 값
        for line in range(lines) :
            index = abs(array[line][0]-value)

            result.append(index)
        idx = result.index(min(result))
    elif flag == 1 : # height 값
        for line in range(lines) :
            index = abs(array[line][1]-value)

            result.append(index)
        idx = result.index(min(result))
    else :
        print("FUCK!")
        
    return idx   

def Detecting(img) :
    img_color = img.copy()
    result = []
    # hsv이미지로 변환한다.
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    #이미지 중앙 값 가져오기.
    width, height = image_center(img_color) 
    #이미지 중앙 hsv값 가져오기.
    hsv_color = img_color[height,width]
    color = hsv_color
    # cvtColor 함수의 입력으로 사용할 수 있도록 한 픽셀로 구성된 이미지로 변환한다. 
    pixel = np.uint8([[color]]) 
    # cvtColor를 사용하여 HSV 색공간으로 변환한다. 
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV) 
    # HSV값을 출력하기위해 픽셀값만 갖고온다. 
    hsv = hsv[0][0] 
    # bgr과 hsv 값 출력 

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    hsv_min, hsv_max = color_range(hsv,40)
    img_mask = cv2.inRange(img_hsv, hsv_min, hsv_max)

    # 원본이미지에서 범위값에 해당하는 영상부분을 흭득한다.
    img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)

    blur = cv2.GaussianBlur(img_result, ksize=(3,3),sigmaX=0)
    
    
    
    canny = cv2.Canny(blur, 10, 255)
    
    rho =2
    theta = np.pi/180
    threshold = 190
    min_line_len =110
    max_line_gap = 150

    lines,point = hough_lines(canny, rho, theta, threshold, 
                       min_line_len, max_line_gap)
    
    idx = nearpos(point,0,1) # 맨위 모서리
    img = cv2.line(img,(point[idx][0],point[idx][1]),(point[idx][0],point[idx][1]),green_color,8)
    result.append(point[idx])
    idx = nearpos(point,height*2,1) # 맨아래 모서리
    img = cv2.line(img,(point[idx][0],point[idx][1]),(point[idx][0],point[idx][1]),green_color,8)
    result.append(point[idx])
    idx = nearpos(point,0,0) # 맨왼쪽 모서리
    img = cv2.line(img,(point[idx][0],point[idx][1]),(point[idx][0],point[idx][1]),green_color,8)
    result.append(point[idx])
    idx = nearpos(point,width*2,0) # 맨오른쪽 모서리
    img = cv2.line(img,(point[idx][0],point[idx][1]),(point[idx][0],point[idx][1]),green_color,8)
    result.append(point[idx])
    result = np.array(result).tolist()
    
    return result

# 이미지 가져오기.
'''
img_color = cv.imread('./testing.jpg')

result,img = Detecting(img_color)

print(result)

plt.figure(figsize=(10,8))
plt.imshow(img)

cv.imshow('Result', img)
cv.waitKey(0)
cv.destroyAllWindows()
'''

