import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

# previous img, current img, middle of bonding box
def track_ssd(pre_img,curr_img,bbox):
    # pre_hsv = cv2.cvtColor(pre_img,cv2.COLOR_BGR2HSV)
    # curr_hsv = cv2.cvtColor(curr_img,cv2.COLOR_BGR2HSV)
    pre_hsv = pre_img
    curr_hsv = curr_img

    # crop face section
    cropped = pre_hsv[bbox[1]:bbox[1]+40, bbox[0]:bbox[0]+40]
    # cv2.imshow('cropped',cropped)
    # cv2.waitKey(0)

    crop_height, crop_width = cropped.shape[:2]
    curr_height, curr_width = curr_hsv.shape[:2]
    # ssd = np.zeros((curr_height - crop_height, curr_width - crop_width))


    min_ssd = np.inf
    for y in range(curr_height - crop_height):
        for x in range(curr_width - crop_width):
            curr_ssd = np.sum((curr_hsv[y:y+crop_height, x:x+crop_width] - cropped)**2)
            
            if curr_ssd<min_ssd:
                # print(curr_ssd,y,x)
                min_ssd = curr_ssd
                min_index = [x,y]

    # min_index = np.unravel_index(np.argmin(ssd), ssd.shape)
    # x, y = min_index[::-1]

    return min_index


def frame_local_ssd(pre_img,curr_img,bbox):

    pre_hsv = pre_img
    curr_hsv = curr_img
    curr_height, curr_width = curr_hsv.shape[:2]

    cropped = pre_hsv[bbox[1]:bbox[1]+40, bbox[0]:bbox[0]+40]

    local_range = (
        max(bbox[1]-20, 0), 
        max(bbox[0]-20, 0), 
        min(bbox[1]+60, curr_height), 
        min(bbox[0]+60, curr_width)
    )
    # print(local_range)

    search_area = curr_hsv[local_range[0]:local_range[2],local_range[1]:local_range[3]]



    crop_height, crop_width = cropped.shape[:2]
    local_height, local_width = search_area.shape[:2]


    min_ssd = np.inf
    for y in range(local_height - crop_height):
        for x in range(local_width - crop_width):
            curr_ssd = np.sum((search_area[y:y+crop_height, x:x+crop_width] - cropped)**2)
            
            if curr_ssd<min_ssd:
                # print(curr_ssd,y,x)
                min_ssd = curr_ssd
                min_index = [x+local_range[1],y+local_range[0]]



    return min_index


def track_local_ssd(directory_path):
    processed_img = []
    filename = '{:04d}.jpg'.format(1)
    filepath = os.path.join(directory_path, filename)
    img_pre = cv2.imread(filepath)
    bbox_center = [50,24]
    bbox_size = 40
    cv2.rectangle(img_pre, (bbox_center[0], bbox_center[1]), (bbox_center[0]+bbox_size, bbox_center[1]+bbox_size), (0, 255, 255), 2)
    processed_img.append(img_pre)
    # cv2.imshow('original',img_pre)
    # cv2.waitKey(0)


    for i in range(2, 500):
        filename = '{:04d}.jpg'.format(i)
        filepath = os.path.join(directory_path, filename)
        img_curr = cv2.imread(filepath)
        # cv2.imshow('original',img_curr)
        bbox_center = frame_local_ssd(img_pre, img_curr, bbox_center)
        cv2.rectangle(img_curr, (bbox_center[0], bbox_center[1]), (bbox_center[0]+bbox_size, bbox_center[1]+bbox_size), (0, 255, 255), 2)
        # cv2.imshow('original',img_curr)
        # cv2.imshow('pre',img_pre)
        # cv2.waitKey(0)
        img_pre = img_curr
        processed_img.append(img_pre)

    return processed_img

def frame_local_cc(pre_img,curr_img,bbox):

    pre_hsv = pre_img
    curr_hsv = curr_img
    curr_height, curr_width = curr_hsv.shape[:2]

    cropped = pre_hsv[bbox[1]:bbox[1]+40, bbox[0]:bbox[0]+40]

    local_range = (
        max(bbox[1]-20, 0), 
        max(bbox[0]-20, 0), 
        min(bbox[1]+60, curr_height), 
        min(bbox[0]+60, curr_width)
    )
    # print(local_range)

    search_area = curr_hsv[local_range[0]:local_range[2],local_range[1]:local_range[3]]



    crop_height, crop_width = cropped.shape[:2]
    local_height, local_width = search_area.shape[:2]


    max_cc = 0
    for y in range(local_height - crop_height):
        for x in range(local_width - crop_width):
            curr_cc = np.sum(search_area[y:y+crop_height, x:x+crop_width]*cropped)
            
            if curr_cc>max_cc:
                # print(curr_ssd,y,x)
                max_cc = curr_cc
                max_index = [x+local_range[1],y+local_range[0]]



    return max_index

def track_local_cc(directory_path):
    processed_img = []
    filename = '{:04d}.jpg'.format(1)
    filepath = os.path.join(directory_path, filename)
    img_pre = cv2.imread(filepath)
    bbox_center = [50,24]
    bbox_size = 40
    cv2.rectangle(img_pre, (bbox_center[0], bbox_center[1]), (bbox_center[0]+bbox_size, bbox_center[1]+bbox_size), (0, 255, 255), 2)
    processed_img.append(img_pre)

    # cv2.imshow('original',img_pre)
    # cv2.waitKey(0)


    for i in range(2, 500):
        filename = '{:04d}.jpg'.format(i)
        filepath = os.path.join(directory_path, filename)
        img_curr = cv2.imread(filepath)

        bbox_center = frame_local_cc(img_pre, img_curr, bbox_center)
        cv2.rectangle(img_curr, (bbox_center[0], bbox_center[1]), (bbox_center[0]+bbox_size, bbox_center[1]+bbox_size), (0, 255, 0), 2)
        
        # cv2.imshow('original',img_curr)
        # cv2.imshow('pre',img_pre)
        # cv2.waitKey(0)
        img_pre = img_curr
        processed_img.append(img_pre)

    return processed_img

def frame_local_ncc(pre_img,curr_img,bbox):

    pre_hsv = pre_img
    curr_hsv = curr_img
    curr_height, curr_width = curr_hsv.shape[:2]

    cropped = pre_hsv[bbox[1]:bbox[1]+40, bbox[0]:bbox[0]+40]

    local_range = (
        max(bbox[1]-20, 0), 
        max(bbox[0]-20, 0), 
        min(bbox[1]+60, curr_height), 
        min(bbox[0]+60, curr_width)
    )
    # print(local_range)

    search_area = curr_hsv[local_range[0]:local_range[2],local_range[1]:local_range[3]]



    crop_height, crop_width = cropped.shape[:2]
    local_height, local_width = search_area.shape[:2]


    max_ncc = 0
    for y in range(local_height - crop_height):
        for x in range(local_width - crop_width):
            I_hat = search_area[y:y+crop_height, x:x+crop_width]-np.mean(search_area[y:y+crop_height, x:x+crop_width])
            T_hat = cropped-np.mean(cropped)

            curr_ncc = np.sum(I_hat*T_hat)/np.sqrt(np.sum(I_hat**2)*np.sum(T_hat**2))
            
            if curr_ncc>max_ncc:
                # print(curr_ssd,y,x)
                max_ncc = curr_ncc
                max_index = [x+local_range[1],y+local_range[0]]



    return max_index

def track_local_ncc(directory_path):
    processed_img = []
    filename = '{:04d}.jpg'.format(1)
    filepath = os.path.join(directory_path, filename)
    img_pre = cv2.imread(filepath)
    bbox_center = [50,24]
    bbox_size = 40
    cv2.rectangle(img_pre, (bbox_center[0], bbox_center[1]), (bbox_center[0]+bbox_size, bbox_center[1]+bbox_size), (0, 255, 255), 2)
    processed_img.append(img_pre)

    # cv2.imshow('original',img_pre)
    # cv2.waitKey(0)


    for i in range(2, 500):
        filename = '{:04d}.jpg'.format(i)
        filepath = os.path.join(directory_path, filename)
        img_curr = cv2.imread(filepath)

        bbox_center = frame_local_ncc(img_pre, img_curr, bbox_center)
        cv2.rectangle(img_curr, (bbox_center[0], bbox_center[1]), (bbox_center[0]+bbox_size, bbox_center[1]+bbox_size), (0, 0, 255), 2)
        
        # cv2.imshow('original',img_curr)
        # cv2.imshow('pre',img_pre)
        # cv2.waitKey(0)
        img_pre = img_curr
        processed_img.append(img_pre)

    return processed_img

def img_to_video(img_list,out_file):

    # out_file = 'output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24.0
    frame_size = (128, 96)

    # Create video writer object
    out = cv2.VideoWriter(out_file, fourcc, fps, frame_size)

    # Iterate over the list of image files
    for image_file in img_list:
        # Write image to video file
        out.write(image_file)

    # Release video writer object
    out.release()


def main():

    path = "image_girl/"

    """ssd"""
    ssd_imgs = track_local_ssd(path)
    out_file_ssd = 'output_ssd.mp4'
    img_to_video(ssd_imgs,out_file_ssd)

    """cc"""
    cc_imgs = track_local_cc(path)
    out_file_cc = 'output_cc.mp4'
    img_to_video(cc_imgs,out_file_cc)

    """ncc"""

    ncc_imgs = track_local_ncc(path)
    out_file_ncc = 'output_ncc.mp4'
    img_to_video(ncc_imgs,out_file_ncc)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()