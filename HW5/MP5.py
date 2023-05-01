import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy import ndimage

from PIL import Image, ImageFilter

"""Canny Edge Detector"""

def CED(img):
    im = np.array(img)
    """smoothing"""
    im = gaussian_filter(im, sigma = 1, truncate = 1)

    """Image Gradient"""
    # y_kernel = [[1,2,1],[0,0,0],[-1,-2,-1]]
    # x_kernel = [[-1,0,1],[-2,0,2],[-1,0,1]]

    # Iy = signal.convolve2d(im, y_kernel)
    # Ix = signal.convolve2d(im, x_kernel)

    # theta = np.arctan(Iy/Ix)

    # print(theta)
    # Mag = np.sqrt(Iy**2+Ix**2)
    # Mag = Mag/np.max(Mag)*255
    theta, Mag = img_gradient(im)
    """suppress non maxima"""
    new_Mag = Suppress_non_Maxima(Mag,theta)
    """generate threshold"""
    thr_Mag_l,thr_Mag_h = threshold_gen(new_Mag,0.7)
    """connect edge"""
    connected = connect_edge(thr_Mag_l,thr_Mag_h)

    fig= plt.figure(figsize=(2,3))
    a = fig.add_subplot(2,3,1)
    a.title.set_text('smoothing')
    plt.imshow(im, cmap='gray')
    b = fig.add_subplot(2,3,2)
    b.title.set_text('Image Gradient')
    plt.imshow(Mag, cmap='gray')
    c = fig.add_subplot(2,3,3)
    c.title.set_text('suppress non maxima')
    plt.imshow(new_Mag, cmap='gray')
    d = fig.add_subplot(2,3,4)
    d.title.set_text('threshold_Low')
    plt.imshow(thr_Mag_l, cmap='gray')
    e = fig.add_subplot(2,3,5)
    e.title.set_text('threshold_High')
    plt.imshow(thr_Mag_h, cmap='gray')
    f = fig.add_subplot(2,3,6)
    f.title.set_text('connect edge')
    plt.imshow(connected, cmap='gray')
    plt.show()
    # fig.show()



    return connected
def img_gradient(img):
    y_kernel = [[1,2,1],[0,0,0],[-1,-2,-1]]
    x_kernel = [[-1,0,1],[-2,0,2],[-1,0,1]]
    Iy = signal.convolve2d(img, y_kernel)
    Ix = signal.convolve2d(img, x_kernel)
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    theta = np.zeros((x,y))

    for i in range(x):
        for j in range(y):
            if Ix[i][j] == 0:
                theta[i][j] = np.pi
            else:
                theta[i][j] = np.arctan(Iy[i][j]/Ix[i][j])
            

    # print(theta)
    Mag = np.sqrt(Iy**2+Ix**2)
    Mag = Mag/np.max(Mag)*255

    return theta, Mag

def Suppress_non_Maxima(img,angle):
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    new_img = np.zeros((x,y))
    deg = angle*180/np.pi
    deg[deg<0]+=180

    for i in range(x):
        for j in range(y):
            try:
                upper = 255
                lower = 255
                #angle 0
                # print(deg)
                if (0 <= deg[i,j] < 22.5) or (157.5 <= deg[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= deg[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= deg[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= deg[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    new_img[i,j] = img[i,j]
                else:
                    new_img[i,j] = 0

            except IndexError as err:
                pass

    return new_img

def threshold_gen(img, precent):
    hist, bin_edges = np.histogram(img, bins=256)
    print(np.shape(hist))
    print(np.shape(bin_edges))
    bin_edges = np.delete(bin_edges,255)
    plt.plot(bin_edges,hist)
    plt.title("Histogram of original image")
    plt.show()

    # generate normalized cumulative histogram
    hist_cumu = np.cumsum(hist)
    hist_normcu = hist_cumu/np.max(hist_cumu)
    plt.plot(bin_edges,hist_normcu)
    plt.title("Cumulative Histogram")
    plt.show()

    high = np.max(hist_normcu)
    low = np.min(hist_normcu)
    print(high)
    print(low)
    limit = (high-low)*precent+low
    print(limit)
    for i in range(np.size(hist_normcu)):
        if hist_normcu[i]>=limit:
            T_high = i
            break
    # print(T_high)
    T_low = T_high/2
    # print(T_low)

    high_val = 255
    low_val = 127
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    new_img_l = np.zeros((x,y))
    new_img_h = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if T_low<=img[i][j]:
                new_img_l[i][j] = high_val
            # elif T_low>img[i][j]:
            #     new_img[i][j] = 0
            # else:
            #     new_img[i][j] = low_val
    for i in range(x):
        for j in range(y):
            if T_high<=img[i][j]:
                new_img_h[i][j] = high_val
    
    return new_img_l,new_img_h

def connect_edge(l_img,h_img):
    x = np.shape(l_img)[0]
    y = np.shape(l_img)[1]
    bonding_box =[[1,1],[1,0],[1,-1],[0,1],[0,-1],[-1,1],[-1,0],[-1,-1]]
    output = np.copy(h_img)
    for i in range(x):
        for j in range(y):
            for dir in bonding_box:
                r = i+dir[0]
                c = j+dir[1]
                if 0<=r<x and 0<=c<y and l_img[i][j]==255 and output[r][c]==255:
                    output[i][j] = 255

    for i in reversed(range(x)):
        for j in reversed(range(y)):
            for dir in bonding_box:
                r = i+dir[0]
                c = j+dir[1]
                if 0<=r<x and 0<=c<y and l_img[i][j]==255 and output[r][c]==255:
                    output[i][j] = 255
    return output

def main():
    # img_1 = cv2.imread("joy1.bmp",cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('original_lena',img_1)
    # imgCED1 = np.uint8(CED(img_1))
    # # print(imgCED1)
    # cv2.imshow('CED_lena',imgCED1)

    img_2 = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original_lena',img_2)
    imgCED2 = np.uint8(CED(img_2))
    # print(imgCED1)
    cv2.imshow('CED_lena',imgCED2)

    """uncomment for other functions"""

    # """Sober"""

    # img_blur = cv2.GaussianBlur(img_2,(3,3), 1)
    # # sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    # # sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)

    # """Roberts"""
    # roberts_cross_v = np.array( [[1, 0 ],
    #                          [0,-1 ]] )
  
    # roberts_cross_h = np.array( [[ 0, 1 ],
    #                             [ -1, 0 ]] )
    # img_r = np.copy(img_2)
    # img_r=img_r/255.0
    # vertical = ndimage.convolve( img_r, roberts_cross_v )
    # horizontal = ndimage.convolve( img_r, roberts_cross_h )
    
    # edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    # edged_img*=255

    # cv2.imshow('roberts function', np.uint8(edged_img))

    # """Crossing Edge"""
    # img = Image.open(r"lena.bmp")
    
    # # Converting the image to grayscale, as Sobel Operator requires
    # # input image to be of mode Grayscale (L)
    # img = img.convert("L")
    
    # # Calculating Edges using the passed laplacian Kernel
    # final = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
    #                                         -1, -1, -1, -1), 1, 0))
    
    # final.save("lena_cross.png")
    # img_3 = cv2.imread("pointer1.bmp",cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('original_lena',img_3)
    # imgCED3 = np.uint8(CED(img_3))
    # # print(imgCED1)
    # cv2.imshow('CED_lena',imgCED3)

    # img_4 = cv2.imread("test1.bmp",cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('original_lena',img_4)
    # imgCED4 = np.uint8(CED(img_4))
    # # print(imgCED1)
    # cv2.imshow('CED_lena',imgCED4)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()