import os
import cv2
import numpy as np

#左上，右上，右下，左下
def img_Transform(pt_lst,img):
    height,width = img.shape[:2]
    matsrc = np.float32([*pt_lst])
    matdst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    matAffine = cv2.getPerspectiveTransform(matsrc,matdst)
    return cv2.warpPerspective(img,matAffine,(width,height))


def match_image(img_path,ignore_flag = True,gray_flag = True,rsz_flag = True):
    cvimg = cv2.imread(img_path)
    h,w = cvimg.shape[:2]

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        cvimg, arucoDict, parameters=arucoParams
    )

    if len(corners) != 4:
        return None if ignore_flag else cvimg

    up_left, up_right, down_right, down_left = (0,0),(0,w),(h,w),(h,0)
    f1,f2,f3,f4 = True,True,True,True

    for i in range(4):
        if ids[i][0] == 0:
            up_left = corners[i][0][2]
            f1 = False
        elif ids[i][0] == 1:
            up_right = corners[i][0][3]
            f2 = False
        elif ids[i][0] == 2:
            down_right = corners[i][0][0]
            f3 = False
        elif ids[i][0] == 3:
            down_left = corners[i][0][1]
            f4 = False
    
    if f1 or f2 or f3 or f4:
        return None if ignore_flag else cvimg
        
    # channels = cvimg.shape[2]

    # mask = np.zeros(cvimg.shape, dtype=np.uint8)
    # ignore_mask_color = (255,) * channels

    # pts_img = np.int32(
    #     [
    #         up_left,
    #         up_right,
    #         down_right,
    #         down_left,
    #     ]
    # )


    # # 无返回值，将mask按照 pts_img 的顺序进行画图
    # cv2.fillPoly(mask, [pts_img], ignore_mask_color)

    # # bitwise_and：将两幅图像进行向与操作
    # masked_image = cv2.bitwise_and(cvimg, mask)

    matsrc = np.float32([up_left,up_right,down_right,down_left])
    matdst = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    matAffine = cv2.getPerspectiveTransform(matsrc,matdst)
    transf_img = cv2.warpPerspective(cvimg,matAffine,(w,h))

    transf_img = cv2.cvtColor(transf_img, cv2.COLOR_RGB2GRAY) if gray_flag else transf_img
    transf_img = cv2.resize(transf_img,(224,224)) if rsz_flag else transf_img

    return transf_img

if __name__ == "__main__":

    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--src", type=str, help='ori_image_path',required=True)
    parse.add_argument("--dst", type=str, help='transfer_image_path')
    parse.add_argument("--gray",action='store_true',default=False)

    args = parse.parse_args()

    if args.dst is None:
        args.dst = os.path.dirname(args.src)

    target_path = os.path.join(args.dst, 'transf_img')
    os.makedirs(target_path,exist_ok=True)
    img_list = os.listdir(args.src)

    from tqdm import tqdm
    for img in tqdm(img_list):
        transf_img = match_image(os.path.join(args.src, img))
        transf_img = cv2.cvtColor(transf_img, cv2.COLOR_RGB2GRAY) if args.gray else transf_img
        # cv2.COLOR_RGB2GRAY 后的图像三通道数值一致
        cv2.imwrite(os.path.join(target_path, img),transf_img)