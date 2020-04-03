import numpy as np
import cv2
from PIL import Image

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
'''
image_path = '/Users/gkalstn/capstone/test_images/img137.jpg'
image = Image.open(image_path)
image = image.resize((845,526))
image_np = load_image_into_numpy_array(image)
'''
def img_warp(total_7_list, image_np):

    mylist = []
    for i in total_7_list:
        mylist.append(tuple(i))

    point_list = mylist[0:4]   # 당구대 좌표 배열
    point_list2 = mylist[4:7]    # 공 좌표 배열

    ##### 한글인식 못함, 에러발생 '빈다이' -> base, '위치2-1' -> test #####
    img_original = image_np # test.jpg 파일을 img_original 변수에 저장
    
    img_original2 = cv2.imread('./trans/base.png') # test3.jpg 파일을 img_original 변수에 저장
    img_original2 = cv2.resize(img_original2, dsize=(650, 344), interpolation=cv2.INTER_AREA)
    

    #(B,G,R)
    cv2.circle(img_original, (point_list2[0]), 2, (255, 255, 255), -1) # R -> W
    cv2.circle(img_original, (point_list2[1]), 2, (0, 0, 255), -1) # B -> R
    cv2.circle(img_original, (point_list2[2]), 2, (0, 255, 255), -1) # G -> Y

    height, width = 315, 612 # return 되는 이미지의 크기 값

    # 좌표 순서 - 상단왼쪽 끝, 상단오른쪽 끝, 하단왼쪽 끝, 하단오른쪽 끝
    pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    print('pts1 : ',pts1)
    print('pts2 : ',pts2)

    M = cv2.getPerspectiveTransform(pts1,pts2)      # pts1의 좌표를 pts2의 좌표로 변환 시킬 변수 M 설정
    a = 1
    b = 2

    img_result = cv2.warpPerspective(img_original, M, (width,height))      # 이미지 와핑
    img_result2 = img_original2

    white=0     # white볼을 둘 공간을 찾았을 때 조건문을 돌리지 않기 위한 변수 R -> W
    red=0     # red볼을 둘 공간을 찾았을 때 조건문을 돌리지 않기 위한 변수 B -> R
    yellow=0     # yellow볼을 둘 공간을 찾았을 때 조건문을 돌리지 않기 위한 변수 G -> Y

    for y in range(15,300):
        for x in range(15,610):
            if img_result[y,x][0] == 255 and img_result[y,x][1] == 255 and img_result[y,x][2] == 255 and white == 0:
                img_result2 = cv2.circle(img_result2, (x+25,y+14),7,(255,255,255),-1)       # 해당 좌표값에 공 그리기
                white = white + 1
                print('<white>\ny, x :',y,', ',x)
            elif img_result[y,x][0] == 0 and img_result[y,x][1] == 255 and img_result[y,x][2] == 255 and yellow == 0:
                img_result2 = cv2.circle(img_result2, (x+25,y+14),7,(0,255,255),-1)
                yellow = yellow + 1
                print('<yellow>\ny, x :',y,', ',x)
            elif img_result[y,x][0] == 0 and img_result[y,x][1] == 0 and img_result[y,x][2] == 255 and red == 0:
                img_result2 = cv2.circle(img_result2, (x+25,y+14),7,(0,0,255),-1)
                red = red + 1
                print('<red>\ny, x :',y,', ',x)
            if white == 1 and red == 1 and yellow == 1:
                k=1
                break
        if white == 1 and red == 1 and yellow == 1:
            break


    print('\nok I know where they are')
    try :
        cv2.imshow("result1", img_original)
        cv2.imshow("result2", img_result2)
        
        ##### 마지막 결과 이미지 저장 #####
        cv2.imwrite("show_result.png", img_result2)
        # cv2.imshow("result3", img_original3)
        cv2.waitKey(0)
    except Exception as e:
        print(str(e))
    # ##### 이미지파일로 리턴 #####
    # return(cv2.imread("show_result.png"))

#img_warp([[256, 500], [736, 328], [225, 302], [571, 0], [225, 137], [385, 141], [539, 141]],image_np)