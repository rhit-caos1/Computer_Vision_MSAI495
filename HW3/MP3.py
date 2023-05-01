import numpy as np
import cv2
import matplotlib.pyplot as plt
#Histogram Equalization function
def HistoEqualization(img):
    im_input = np.array(img)
    im = im_input.flatten()
    print(im)

    # generate histogram for original image
    hist, bin_edges = np.histogram(im, bins=256)
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

    # transfer original image to a new image according to normalized cumulative histogram
    x = np.shape(im_input)[0]-1
    y = np.shape(im_input)[1]-1
    new_img = np.zeros((x+1,y+1))
    for i in range(x+1):
        for j in range(y+1):
            new_img[i][j] = int(hist_normcu[int(im_input[i][j])]*255)

    # generate histogram for processed image
    flat_new_img = new_img.flatten()
    hist, bin_edges = np.histogram(flat_new_img, bins=np.arange(256))
    print(np.shape(hist))
    print(np.shape(bin_edges))
    bin_edges = np.delete(bin_edges,255)
    plt.plot(bin_edges,hist)
    plt.title("Histogram of processed image")
    plt.show()
    return new_img

def main():
    img_1 = cv2.imread("moon.bmp",cv2.IMREAD_GRAYSCALE)

    cv2.imshow('original',img_1)


    img_d1 = np.uint8(HistoEqualization(img_1))
    cv2.imshow('updated',img_d1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
