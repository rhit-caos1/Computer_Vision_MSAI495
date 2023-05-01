import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

"""Function to Generate 2d Histogram for Hue and Saturation"""
def GenHisto2d(H,S):
    Hist, xedges, yedges = np.histogram2d(H, S, bins=(np.arange(257), np.arange(257)),density = True)
    Hist = Hist/(np.amax(Hist))
    plt.imshow(Hist)
    plt.colorbar()
    plt.show()
    np.save("Hist.npy", Hist)
    return Hist

"""Function to seperate skin color from background"""
def seperate_skin(img,Hist):
    im_input = np.array(img)
    x = np.shape(im_input)[0]-1
    y = np.shape(im_input)[1]-1
    
    for i in range(x+1):
        for j in range(y+1):
            H = im_input[i][j][0]
            S = im_input[i][j][1]
            # if H == 255:
            #     H = 254
            # if S == 255:
            #     S = 254
            if Hist[H][S]<0.1:
                im_input[i][j][0] = 0
                im_input[i][j][1] = 0
                im_input[i][j][2] = 0
    return im_input

"""Get Hue and Saturation from the image"""
def GetHS(img,H,S):
    im_input = np.array(img)
    x = np.shape(im_input)[0]-1
    y = np.shape(im_input)[1]-1

    for i in range(x+1):
        for j in range(y+1):
            if im_input[i][j][1]>30:
                H.append(im_input[i][j][0])
                S.append(im_input[i][j][1])

def GetHSfromHLS(img,H,S):
    im_input = np.array(img)
    x = np.shape(im_input)[0]-1
    y = np.shape(im_input)[1]-1

    for i in range(x+1):
        for j in range(y+1):
            if im_input[i][j][2]>30 and im_input[i][j][2]<250 and im_input[i][j][0]<50:
                H.append(im_input[i][j][0])
                S.append(im_input[i][j][2])

def main():
    img_1 = cv2.imread("gun1.bmp")
    cv2.imshow('original',img_1)
    imgHSV = cv2.cvtColor(img_1,cv2.COLOR_BGR2HSV)

    """Training function, comment it out if a Hist matrix is created"""
    # directory_path = "Hand/" # make sure to put the 'r' in front
    # filepaths  = glob.glob(os.path.join(directory_path, "*.jpg"))

    # H = []
    # S = []

    # for path in filepaths:
    #     img_1 = cv2.imread(path)
    #     imgHSV = cv2.cvtColor(img_1,cv2.COLOR_BGR2HSV)
    #     GetHS(imgHSV,H,S)

    # Hist = GenHisto2d(H,S)

    """Test function"""

    Hist = np.load("Hist.npy")
    # print(Hist)
    plt.imshow(Hist)
    plt.colorbar()
    plt.show()
    img_1 = cv2.imread("gun1.bmp")
    cv2.imshow('originalgun',img_1)
    imgHSV1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2HSV)
    imgHSV1 = cv2.cvtColor(seperate_skin(imgHSV1,Hist),cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV_gun',imgHSV1)

    img_2 = cv2.imread("joy1.bmp")
    cv2.imshow('originaljoy',img_2)
    imgHSV2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2HSV)

    imgHSV2= cv2.cvtColor(seperate_skin(imgHSV2,Hist),cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV_joy',imgHSV2)

    img_3 = cv2.imread("pointer1.bmp")
    cv2.imshow('originalpoint',img_3)
    imgHSV3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2HSV)

    imgHSV3 = cv2.cvtColor(seperate_skin(imgHSV3,Hist),cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV_point',imgHSV3)

# HLS color space
    # img_1 = cv2.imread("gun1.bmp")
    # cv2.imshow('original',img_1)
    # imgHLS = cv2.cvtColor(img_1,cv2.COLOR_BGR2HLS)
    # cv2.imshow('originalHLS',imgHLS)

    # directory_path = "Hand/" # make sure to put the 'r' in front
    # filepaths  = glob.glob(os.path.join(directory_path, "*.jpg"))

    # H = []
    # S = []
    # # GetHSfromHLS(imgHLS,H,S)
    # # Hist = GenHisto2d(H,S)

    # for path in filepaths:
    #     img_1 = cv2.imread(path)
    #     imgHLS = cv2.cvtColor(img_1,cv2.COLOR_BGR2HLS)
    #     GetHSfromHLS(imgHLS,H,S)

    # Hist = GenHisto2d(H,S)

    # """Test function"""

    # Hist = np.load("Hist.npy")
    # plt.imshow(Hist)
    # plt.colorbar()
    # plt.show()
    # img_1 = cv2.imread("gun1.bmp")
    # cv2.imshow('original',img_1)
    # imgHSL1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2HLS)
    # imgHSL1 = cv2.cvtColor(seperate_skin(imgHSL1,Hist),cv2.COLOR_HLS2BGR)
    
    
    # cv2.imshow('HSV_gun',imgHSL1)

    #cv show

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()