import cv2


def show_result(orgin_png) :

    left = cv2.imread(orgin_png)    # left = cv2.imread("위치2-1.jpeg") #입력받은 초기 이미지
    right = cv2.imread("./test_image_result/show_result.png")   # 처리 완료한 결과 이미지
    left = cv2.resize(left, dsize=(344, 650), interpolation=cv2.INTER_AREA)

    result = cv2.hconcat([left, right])     # 두 이미지 합친 마지막 결과
    cv2.imshow("last result", result)
    cv2.imwrite("last_result.jpg", result)
    cv2.waitKey(0)

