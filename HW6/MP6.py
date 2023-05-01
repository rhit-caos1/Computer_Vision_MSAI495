import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans

from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy import ndimage

from PIL import Image, ImageFilter

"""Canny Edge Detector"""

def CED(img,sigma_in,truncate_in,threshold_percent):
    im = np.array(img)
    """smoothing"""
    im = gaussian_filter(im, sigma = sigma_in, truncate = truncate_in)

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
    thr_Mag_l,thr_Mag_h = threshold_gen(new_Mag,threshold_percent)
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

# def Hough_trans(img,clusters,threshold,raw_img):
#     x = np.shape(img)[0]
#     y = np.shape(img)[1]
#     points = []
#     for i in range(x):
#         for j in range(y):
#             if img[i][j]==255:
#                 points.append([i,j])
#     lenght = len(points)
#     print(points[0])
#     t_size = 1800
#     r_size = round(np.sqrt(x**2+y**2)+0.5)*2*2+1
#     # votemap_t = []
#     # votemap_r = []
#     votemap = np.zeros([t_size,r_size])
#     for i in range(lenght):
#         for theta in range(t_size):
#             # q = round(points[i][0]*np.cos((theta/180.0)*np.pi)+points[i][1]*np.sin((theta/180.0)*np.pi))
#             q = round((points[i][0]*np.cos((theta/t_size)*np.pi)+points[i][1]*np.sin((theta/t_size)*np.pi))*2)+round(np.sqrt(x**2+y**2)*2+0.5)
#             # votemap_t.append(theta)
#             # votemap_r.append(q)
#             votemap[theta][q] +=1
#     # print(np.sum(votemap))
#     votemap = votemap/(np.max(votemap))*255
#     plt.imshow(votemap, cmap='gray')
#     # plt(np.array(votemap_t),np.array(votemap_r))
#     plt.show()

#     # generate histogram for original image
#     hist, bin_edges = np.histogram(votemap, bins=256)
#     print(np.shape(hist))
#     print(np.shape(bin_edges))
#     bin_edges = np.delete(bin_edges,255)
#     plt.plot(bin_edges,hist)
#     plt.title("Histogram of original image")
#     plt.show()

#     # generate normalized cumulative histogram
#     hist_cumu = np.cumsum(hist)
#     hist_normcu = hist_cumu/np.max(hist_cumu)*255
#     plt.plot(bin_edges,hist_normcu)
#     plt.title("Cumulative Histogram")
#     plt.show()

#     # transfer original image to a new image according to normalized cumulative histogram
#     # x = t_size
#     # y = r_size
#     new_img = np.zeros((t_size,r_size))
#     for i in range(t_size):
#         for j in range(r_size):
#             new_img[i][j] = int(hist_normcu[int(votemap[i][j])])

#     # votemap = votemap/(np.max(votemap))*200
#     # votemap[votemap>0] += 55
#     plt.imshow(new_img, cmap='gray')
#     # plt(np.array(votemap_t),np.array(votemap_r))
#     plt.show()

#     votemap[votemap<threshold] = 0
#     plt.imshow(votemap, cmap='gray')
#     # plt(np.array(votemap_t),np.array(votemap_r))
#     plt.show()
#     coordinate = []
#     for i in range(t_size):
#         for j in range(r_size):
#             if votemap[i][j]>0:
#                 coordinate.append([i,j])

#     # print(coordinate)
    
#     kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(np.array(coordinate))
#     centers = kmeans.cluster_centers_
#     print(centers)

#     plt.imshow(new_img, cmap='gray')
#     plt.scatter(centers[:,1],centers[:,0])
#     plt.show()

#     raw_x = np.shape(raw_img)[0]
#     raw_y = np.shape(raw_img)[1]

#     for point in centers:
#         p = (point[1]-r_size/2)/2
#         t = point[0]/1800.0*np.pi
#         print("theta =",t)
#         print("rho =",p)
#         for i in range(raw_x):
#             j = round((p-i*np.cos(t))/np.sin(t))
#             if 0<=j<raw_y and 0<=i<raw_x:
#                 raw_img[i][j]=255

#     return raw_img

def Hough_trans_new(img,clusters,threshold,raw_img, param):
    x = np.shape(img)[0]
    y = np.shape(img)[1]
    points = []
    for i in range(x):
        for j in range(y):
            if img[i][j]==255:
                points.append([i,j])
    lenght = len(points)
    print(points[0])
    t_size = int(180*param)
    r_size = round(round(np.sqrt(x**2+y**2)+0.5)*2*param/5+1)

    votemap = np.zeros([t_size,r_size])
    for i in range(lenght):
        for theta in range(t_size):

            q = round((points[i][0]*np.cos((theta/t_size)*np.pi)+points[i][1]*np.sin((theta/t_size)*np.pi))*param/5)+round(np.sqrt(x**2+y**2)*param/5+0.5)

            votemap[theta][q] +=1

    votemap = votemap/(np.max(votemap))*255
    plt.imshow(votemap, cmap='gray')

    plt.show()

    # generate histogram for original image
    hist, bin_edges = np.histogram(votemap, bins=256)
    print(np.shape(hist))
    print(np.shape(bin_edges))
    bin_edges = np.delete(bin_edges,255)
    plt.plot(bin_edges,hist)
    plt.title("Histogram of original image")
    plt.show()

    # generate normalized cumulative histogram
    hist_cumu = np.cumsum(hist)
    hist_normcu = hist_cumu/np.max(hist_cumu)*255
    plt.plot(bin_edges,hist_normcu)
    plt.title("Cumulative Histogram")
    plt.show()

    # transfer original image to a new image according to normalized cumulative histogram

    new_img = np.zeros((t_size,r_size))
    for i in range(t_size):
        for j in range(r_size):
            new_img[i][j] = int(hist_normcu[int(votemap[i][j])])


    plt.imshow(new_img, cmap='gray')

    plt.show()

    votemap[votemap<threshold] = 0
    plt.imshow(votemap, cmap='gray')

    plt.show()
    coordinate = []
    for i in range(t_size):
        for j in range(r_size):
            if votemap[i][j]>0:
                coordinate.append([i,j])


    
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(np.array(coordinate))
    centers = kmeans.cluster_centers_
    print(centers)

    plt.imshow(new_img, cmap='gray')
    plt.scatter(centers[:,1],centers[:,0])
    plt.show()

    raw_x = np.shape(raw_img)[0]
    raw_y = np.shape(raw_img)[1]

    for point in centers:
        p = (point[1]-r_size/2)/param*5
        t = point[0]/(180*param)*np.pi
        print("theta =",t)
        print("rho =",p)
        for i in range(raw_x):
            j = round((p-i*np.cos(t))/np.sin(t))
            if 0<=j<raw_y and 0<=i<raw_x:
                raw_img[i][j]=255

    return raw_img


def main():
    img_1 = cv2.imread("test.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original_input',img_1)
    imgCED1 = np.uint8(CED(img_1,1,1,0.7))

    cv2.imshow('test1',imgCED1)
    
    test1 = np.uint8(Hough_trans_new(imgCED1,clusters=4,threshold=100,raw_img = img_1,param = 10))
    cv2.imshow('test1_out',test1)

    img_2 = cv2.imread("test2.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original_input2',img_2)
    imgCED2 = np.uint8(CED(img_2,1,1,0.7))

    cv2.imshow('test2',imgCED2)
    
    test2 = np.uint8(Hough_trans_new(imgCED2,clusters=6,threshold=100,raw_img = img_2, param = 10))
    cv2.imshow('test2_out',test2)

    img_3 = cv2.imread("input.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original_input3',img_3)
    imgCED3 = np.uint8(CED(img_3,1,1,0.94))

    imgCED3[:,0] = 0
    imgCED3[:,1] = 0
    imgCED3[0,:] = 0
    imgCED3[1,:] = 0
    cv2.imshow('test3',imgCED3)
    
    test3 = np.uint8(Hough_trans_new(imgCED3,clusters=5,threshold=80, raw_img = img_3, param = 10))
    cv2.imshow('test3_out',test3)

    """different Param space"""
    """space: 900 x sqrt(r^2+c^2)"""
    img_4 = cv2.imread("input.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original_input',img_4)
    imgCED4 = np.uint8(CED(img_4,1,1,0.94))
    cv2.imshow('test4',imgCED4)

    imgCED4[:,0] = 0
    imgCED4[:,1] = 0
    imgCED4[0,:] = 0
    imgCED4[1,:] = 0
    
    test4 = np.uint8(Hough_trans_new(imgCED4,clusters=5,threshold=80,raw_img = img_4,param=5))
    cv2.imshow('test4_out',test4)

    """space: 4500 x sqrt(r^2+c^2)*5"""
    img_5 = cv2.imread("input.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original_input',img_5)
    imgCED5 = np.uint8(CED(img_5,1,1,0.94))
    cv2.imshow('test5',imgCED5)
    
    imgCED5[:,0] = 0
    imgCED5[:,1] = 0
    imgCED5[0,:] = 0
    imgCED5[1,:] = 0

    test5 = np.uint8(Hough_trans_new(imgCED5,clusters=5,threshold=75,raw_img = img_5,param=15))
    cv2.imshow('test5_out',test5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()