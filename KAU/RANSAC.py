import numpy as np
import matplotlib.pyplot as plt
import cv2 
import sklearn
from tqdm.notebook import tqdm
from numpy.linalg import inv

### (x,y), (x',y') 총 4쌍 (parameter = 8개) ###
def random_matching_point(num, key1, des1, key2, des2):
    cordi = []
    warp_cordi = []

    for i in range(len(key1)):
        cordi.append([*key1[i].pt])

    for i in range(len(key2)):
        warp_cordi.append([*key2[i].pt])

    cordi = np.array(cordi)
    warp_cordi = np.array(warp_cordi)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
    matches = matcher.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:80]

    Select_sample = np.random.randint(0,len(good_matches), size=num)

    queryIdx = []
    trainIdx = []
    distance = []

    for i in range(len(Select_sample)):
        queryIdx.append(good_matches[Select_sample[i]].queryIdx)
        trainIdx.append(good_matches[Select_sample[i]].trainIdx)
        distance.append(good_matches[Select_sample[i]].distance)
    
    sample = np.c_[cordi[queryIdx,0], cordi[queryIdx,1]]
    target = np.c_[warp_cordi[trainIdx,0], warp_cordi[trainIdx,1]]
    return sample, target

### perspective matrix ###
def matrix_8x8(point1, point2):
    point1_x = point1[:,0]
    point1_y = point1[:,1]
    point2_x = point2[:,0]
    point2_y = point2[:,1]

    y = np.array([point2_x[0], point2_y[0], point2_x[1], point2_y[1], point2_x[2], point2_y[2], point2_x[3], point2_y[3]])
    y = y[:,np.newaxis]
    
    arr1 = [point1_x[0], point1_y[0], 1, 0, 0, 0, -point1_x[0]*point2_x[0], -point2_x[0]*point1_y[0]]
    arr2 = [0, 0, 0, point1_x[0], point1_y[0], 1, -point1_x[0]*point2_y[0], -point2_y[0]*point1_y[0]]
    arr3 = [point1_x[1], point1_y[1], 1, 0, 0, 0, -point1_x[1]*point2_x[1], -point2_x[1]*point1_y[1]]
    arr4 = [0, 0, 0, point1_x[1], point1_y[1], 1, -point1_x[1]*point2_y[1], -point2_y[1]*point1_y[1]]
    arr5 = [point1_x[2], point1_y[2], 1, 0, 0, 0, -point1_x[2]*point2_x[2], -point2_x[2]*point1_y[2]]
    arr6 = [0, 0, 0, point1_x[2], point1_y[2], 1, -point1_x[2]*point2_y[2], -point2_y[2]*point1_y[2]]
    arr7 = [point1_x[3], point1_y[3], 1, 0, 0, 0, -point1_x[3]*point2_x[3], -point2_x[3]*point1_y[3]]
    arr8 = [0, 0, 0, point1_x[3], point1_y[3], 1, -point1_x[3]*point2_y[3], -point2_y[3]*point1_y[3]]

    arr = np.vstack([arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8])
    return y, arr

### perspective matrix로 img를 사영 (alery system O) ###
def projection_img(matrix, img):
    Alert = 0
    out_img = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            theta = img[y,x]
            result = matrix @ np.array([[x, y, 1]]).T

            if (int(result[0]) >= 270) or (int(result[1]) >= 270) or (int(result[0]) < 0) or (int(result[1]) < 0):
                Alert = 1 
                break;

            out_img[int(result[1]),int(result[0])] = theta # Perspective Projection한 image
        ### 활성화한 Alert이 있다면 img 생성 중지, 생성된 img는 interpolation이 생긴 형태가 나와 알고리즘에 error 발생###
        if Alert == 1: 
            Alert = 0 
            break;  
    
    return out_img

### projected img에 대한 서술자 생성 ###
def ransac_matching_point(ransac_kps, ransac_dsr, warp_kps, warp_dsr, matcher):
    gt_cordi = [] # g.t keypoint의 좌표
    ransac_cordi = [] # 추정 keypoint의 좌표

    for i in range(len(ransac_kps)):
        ransac_cordi.append([*ransac_kps[i].pt])
    for i in range(len(warp_kps)):
        gt_cordi.append([*warp_kps[i].pt])

    gt_cordi = np.array(gt_cordi)
    ransac_cordi = np.array(ransac_cordi)

    ransac_matches = matcher.match(warp_dsr, ransac_dsr)

    ransac_matches = sorted(ransac_matches, key=lambda x: x.distance) # mathching point를 유사도에 따라 sorting시킴
    good_ransac_matches = ransac_matches[:80]

    ransac_queryIdx = [] # g.t 서술자의 matching point의 index를 나타냄
    ransac_trainIdx = [] # 추정 서술자의 matching point의 index를 나타냄
    ransac_distance = [] # 두 matching point 간의 유사도를 나타냄

    for i in range(len(good_ransac_matches)):
        ransac_queryIdx.append(good_ransac_matches[i].queryIdx)
        ransac_trainIdx.append(good_ransac_matches[i].trainIdx)
        ransac_distance.append(good_ransac_matches[i].distance)

    gt_key = np.c_[gt_cordi[ransac_queryIdx,0], gt_cordi[ransac_queryIdx,1]]
    ransac_key = np.c_[ransac_cordi[ransac_trainIdx,0], ransac_cordi[ransac_trainIdx,1]]
    return gt_key, ransac_key, ransac_distance


imagefile = r"/content/drive/MyDrive/항공대학교/2023_1학기_강의자료/프로젝트/영상처리/final/image/burgerking(old)_logo.png"

img = cv2.imread(imagefile)
img = img[5:275,50:320]
print(img.shape)
color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perspective Transform 생성
height, width = gray.shape
warp_x, warp_y = 100, 200
rand_x, rand_y = np.random.randint(0,100,size=2)

origin_point = np.array([[0,0],[0,width],[height,0],[height,width]], dtype=np.float32)
wrap_point = np.array([[25,25], [25,width-rand_x], [height-rand_y, 5], [height-rand_y,width-rand_x]], dtype=np.float32)

Matrix = cv2.getPerspectiveTransform(origin_point, wrap_point)
warp_img = cv2.warpPerspective(gray, Matrix, (height,width))

sift = cv2.xfeatures2d.SIFT_create()
kps,dsr = sift.detectAndCompute(gray, None)
warp_kps,warp_dsr = sift.detectAndCompute(warp_img, None)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
matches = matcher.match(dsr, warp_dsr)

matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:40]

res = cv2.drawMatches(gray, kps, warp_img, warp_kps, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(res, cmap='gray')
plt.axis('off')

######################## RANSAC algorithm (alert system o) ########################

iter = 100
best_Idx = []
best_num = []
best_params = []
best_gt_key = []
best_ransac_key = []

for i in tqdm(range(iter)):
    sample, target = random_matching_point(4, kps, dsr, warp_kps, warp_dsr)

    y_warp, arr_point = matrix_8x8(sample, target) ## Perspective matrix => output = 8x1, input = 8x8 => parameter = 8x1

    if np.linalg.det(arr_point) == 0: ## 역행렬이 불가능 할 경우 iteration skip
        continue;

    theta = inv(arr_point) @ y_warp
    theta_arr = np.append(theta, np.array([1])).reshape(3,3)

    projected_img = projection_img(theta_arr, gray)
            
    ransac_kps, ransac_dsr = sift.detectAndCompute(projected_img, None) ## image에 대한 SIFT 서술자 생성

    if ransac_dsr is not None:
        ### 추청한 parameter로 inlier 매기는 작업 시작 ###
        gt_key, ransac_key, ransac_distance = ransac_matching_point(ransac_kps, ransac_dsr, warp_kps, warp_dsr, matcher)

        ransac_distance = (ransac_distance-np.min(ransac_distance)) / (np.max(ransac_distance)-np.min(ransac_distance)+1e-7) ## 두 매칭점의 유사도를 min-max 정규화로 구현
        distance_Idx = np.where(ransac_distance < 0.3)[0] ## 문턱값을 넘긴 값을 index로 넘김 (유사도가 높은 distance의 인덱스를 반환)

        ### 문턱값을 넘긴 경우 ###
        if len(distance_Idx) > 10:
            best_num.append(len(distance_Idx)) # 문턱값을 넘긴 Idx의 개수
            best_Idx.append(distance_Idx) # 문턱값을 넘긴 Idx => matching이 잘된 keypoint에 대한 index
            best_params.append(theta_arr) # 문턱값을 넘긴 parameter
            best_gt_key.append(gt_key) # 문턱값을 넘긴 g.t keypoint의 개수를 저장
            best_ransac_key.append(ransac_key) # 문턱값을 넘긴 추정 keypoint의 개수를 저장

    else: # 서술자가 형성되지 않은 경우
        continue;


t = best_num.index(max(best_num))

ransac_key = best_ransac_key[t]
gt_key = best_gt_key[t]

img = np.zeros_like(gray)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        theta = gray[y,x]
        result = best_params[t] @ [x, y, 1]
        if (int(result[0]) >= 270) or (int(result[1]) >= 270) or (int(result[0]) < 0) or (int(result[1]) < 0):
                continue;
        img[int(result[1]),int(result[0])] = theta

inlier_ratio_alertO = max(best_num)/len(gt_key)

# plt.suptitle(f'Inlier ratio among matching points: {inlier_ratio_alertO}%', fontsize = 15)
plt.subplot(1,2,1)
plt.title('ransac')
plt.imshow(img,cmap='gray')
plt.plot(ransac_key[best_Idx[t],0],ransac_key[best_Idx[t],1],'ro', markersize = 2)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('target')
plt.imshow(warp_img,cmap='gray')
plt.plot(gt_key[best_Idx[t],0],gt_key[best_Idx[t],1],'ro', markersize = 2)
plt.axis('off')
plt.tight_layout()
