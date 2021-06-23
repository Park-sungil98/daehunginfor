import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract

"""이미지 생성"""
img_ori = cv2.imread('car2.png')

height, width, channel = img_ori.shape
plt.figure(figsize=(12, 10))
"""이미지 출력 및 출력할 화면의 색깔 설정"""
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
"""
차량 번호판을 확실하게 인식하기 위하여 흑백 화면으로 만들기
가우시안 블러 적용
"""
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_blur_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
"""
Countours란 동일한 색 또는 동일한 강도를 가지고 있는 영역의 
경계선을 연결한 선
findcontours를 잉요하여 검은 색 바탕에서 흰색 대상을 찾는다
때문에 위에서 흑백으로 바꿔준 것
"""
contours,_= cv2.findContours(
    img_blur_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

plt.figure(figsize=(12, 10))
plt.imshow(temp_result)
plt.show()




