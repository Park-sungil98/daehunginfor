import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract

"""이미지 생성"""
img_ori = cv2.imread('car2.png')

height, width, channel = img_ori.shape
"""이미지 출력 및 출력할 화면의 색깔 설정"""
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
"""
차량 번호판을 확실하게 인식하기 위하여 흑백 화면으로 만들기
가우시안 블러 적용
"""
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
"""
Contours란 동일한 색 또는 동일한 강도를 가지고 있는 영역의 
경계선을 연결한 선
findcontours를 잉요하여 검은 색 바탕에서 흰색 대상을 찾는다
때문에 위에서 흑백으로 바꿔준 것
"""
contours,_= cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)
"""
Contours들의 좌표를 이용하여 사각형 형태로 그리고 
딕셔너리를 하나 생성하여
contours들의 정보를 저장한다.
"""
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict=[]

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255), thickness=2)
    contours_dict.append({
        'contour':contour,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'cx': x +(w/2),
        'cy': y+(h/2)
    })

# cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

"""
비율을 이용하여 번호판 글자 인 것 같은 contours를 추려냄
최소 비율을 0.25, 최대 비율을 1.0으로 설정하여 너비와 높이를
이용하여 비율을 구하고 기준에 맞는 contours만 저장
"""

MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours=[]

cnt =0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w']/d['h']

    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt+=1
        possible_contours.append(d)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result,pt1=(d['x'],d['y']), pt2=(d['x']+d['w'],d['y']+d['h']), color=(255, 255, 255), thickness=2)
#contour와 다른 countour의 대가선의 5배 안에 있어야함
MAX_DIAG_MULTIPLYER = 5
#contour와 다른 countour의 각도(세타)의 크기 지정
MAX_ANGLE_DIFF = 12.0
#면적 차이
MAX_AREA_DIFF = 0.5
#너비차이
MAX_WIDTH_DIFF = 0.8
#높이 차이
MAX_HEIGHT_DIFF = 0.2
#위의 조건을 만족하는 값들이 3개 미만이면 옳지 않음
MIN_N_MATCHED = 3

#나중에 재귀로 계속 찾아야되기 때문에 함수로 지정
def find_chars(contour_list):
    #최종적으로 남는 인덱스 값들을 저장
    matched_result_idx = []

    #contour하나와 또 다른 contour들을 비교하기 위하여 for문을 2번 돌게됨
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            #두 중심의 거리 구하기
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            #각도 구하기
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

PLATE_WIDTH_PADDING = 1.3  # 1.3
PLATE_HEIGHT_PADDING = 1.5  # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )



    plt.subplot(len(matched_result), 1, i + 1)
    plt.imshow(img_cropped, cmap='gray')
    plt.show()



