#import numpy as np
import cv2
import math
#


def transcord(h, w, ball_point, billiard):

    a = ball_point[0]
    b = ball_point[1]

    a2 = (billiard[2][0] - a) * (billiard[3][1] - billiard[2][1]) - \
         (billiard[2][1] - b) * (billiard[3][0] - billiard[2][0])
    b2 = (billiard[2][0] - a) * (billiard[1][1] - billiard[0][1]) + (billiard[0][0] - a) * \
         (billiard[3][1] - billiard[2][1]) - (billiard[2][1] - b) * (billiard[1][0] - billiard[0][0]) - \
         (billiard[0][1] - b) * (billiard[3][0] - billiard[2][0])
    c = (billiard[0][0] - a) * (billiard[1][1] - billiard[0][1]) - (billiard[0][1] - b) * \
        (billiard[1][0] - billiard[0][0])

    k = b2**2 - 4*a2*c
    
    if k > 0:
        m = (b2*(-1) + math.sqrt(k)) / (2*a2)
        if m < 0:
            m = (b2 * (-1) - math.sqrt(k)) / (2 * a2)

    a2 = (billiard[1][0] - a) * (billiard[3][1] - billiard[1][1]) - (billiard[1][1] - b) * \
         (billiard[3][0] - billiard[1][0])
    b2 = (billiard[1][0] - a) * (billiard[2][1] - billiard[0][1]) + (billiard[0][0] - a) * \
         (billiard[3][1] - billiard[1][1]) - (billiard[1][1] - b) * (billiard[2][0] - billiard[0][0]) - \
         (billiard[0][1] - b) * (billiard[3][0] - billiard[1][0])
    c = (billiard[0][0] - a) * (billiard[2][1] - billiard[0][1]) - (billiard[0][1] - b) * \
        (billiard[2][0] - billiard[0][0])
    
    k = b2**2 - 4*a2*c

    if k > 0:
        n = (b2*(-1) + math.sqrt(k)) / (2*a2)
        if n < 0:
            n = (b2*(-1) - math.sqrt(k)) / (2*a2)

    h2 = h * m / (m+n)
    w2 = w * n / (m+n)

    return (w2, h2)



def warp(array):

    img_original2 = cv2.imread('./trans/base.jpeg')  # test3.jpg 파일을 img_original 변수에 저장

    img_result2 = cv2.resize(img_original2, dsize=(348,630), interpolation=cv2.INTER_AREA)

#    width, height = 315, 612  # return 되는 이미지의 크기 값
    height, width = 612, 315
    
    '''
    # 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
    pts1 = np.float32([list(array[0]), list(array[1]), list(array[2]), list(array[3])])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)  # pts1의 좌표를 pts2의 좌표로 변환 시킬 변수 M 설정
    #print(M)
#    print('왼상단 : ', np.dot(M, pts1[0]))
#    print('우상단 : ', np.dot(M, pts1[1]))
#    print('왼하단 : ", np.dot(M, pts1[2]))
#    print("왼하단 : ", np.dot(M, pts1[3]))
    ball_list = []

    brr = []
    for i in range(4,7):
        temp = [array[i][0], array[i][1], 1]
        temp = np.reshape(temp, (3, 1))

        brr.append(np.dot(M, temp))
        x = brr[i-4][0] / brr[i-4][2]# + 18
        y = brr[i - 4][1] / brr[i - 4][2]# + 18
        if array[i][0] == 0 and array[i][1] == 0:
            x, y = -10, -10
        ball_list.append((x, y))

    '''
    
    billiard = array[:4]
    ball_point = array[4:]

    ball_list = []
    ball_list.append(transcord(height, width, ball_point[0], billiard ))
    ball_list.append(transcord(height, width, ball_point[1], billiard ))
    ball_list.append(transcord(height, width, ball_point[2], billiard ))
    
    print(ball_list)
    img_result2 = cv2.circle(img_result2, (int(ball_list[0][0]), int(ball_list[0][1])), 10, (255, 255, 255), -1)  # 해당 좌표값에 공 그리기
    img_result2 = cv2.circle(img_result2, (int(ball_list[1][0]), int(ball_list[1][1])), 10, (0, 0, 255), -1)
    img_result2 = cv2.circle(img_result2, (int(ball_list[2][0]), int(ball_list[2][1])), 10, (0, 255, 255), -1)

    #return cv2.imshow("result2", img_result2)

#    cv2.imwrite("./test_image_result/test_img_result.png", img_result2)
    return img_result2
# warp([(360, 683), (805, 554), (9, 310), (322, 312), (241, 518), (444, 487), (595, 466)])
# warp([(23, 464), (817, 444), (226, 94), (525, 98), (226, 142), (384, 142), (539, 142)])
# warp([(3, 346), (558, 501), (488, 85), (795, 99), (454, 156), (572, 139), (0,0)])          # 위치 1-4
# warp([(236, 668), (827, 527), (26, 290), (347, 281), (356, 533), (417, 458), (468, 404)])
# warp([(17, 533), (474, 692), (523, 365), (821, 393), (241, 475), (533, 606), (690, 481)])     # 위치 3-3
# warp([(60, 533), (598, 654), (510, 355), (810, 368), (272, 467), (444, 491), (660, 525)])       # 위치 2-4

