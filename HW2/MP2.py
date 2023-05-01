import numpy as np
import cv2

def Erosion(img, SE):
    im = np.array(img)
    # im = np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])
    x = np.shape(im)[0]-1
    y = np.shape(im)[1]-1
    new_img = np.zeros((x+1,y+1))
    for i in range(x+1):
        for j in range(y+1):
            check = True
            for dir in SE:
                next = [i+dir[0],j+dir[1]]
                if 0<=next[0]<=x and 0<=next[1]<=y and im[next[0]][next[1]]==0:
                    check = False
                    break
            if check == True:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0
            
    return new_img

def Dilation(img, SE):
    im = np.array(img)

    x = np.shape(im)[0]-1
    y = np.shape(im)[1]-1
    new_img = np.zeros((x+1,y+1))
    for i in range(x+1):
        for j in range(y+1):
            if 0<=i<=x and 0<=j<=y and im[i][j] == 255:

                for dir in SE:
                    new = [i+dir[0],j+dir[1]]
                    if 0<=new[0]<=x and 0<=new[1]<=y:

                        new_img[new[0]][new[1]]=255

            
    return new_img

def Opening(img, SE):
    return Dilation(Erosion(img,SE),SE)

def Closing(img, SE):
    return Erosion(Dilation(img,SE),SE)

def Boundary(img):
    SE = [(1,1),(1, 0), (1, -1), (0, 1), (0, 0),(0,-1),(-1,1),(-1,0),(-1,-1)]
    erosion_img = Erosion(img,SE)
    return img-erosion_img


def main():
    img_1 = cv2.imread("palm.bmp",cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread("gun.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('palm',img_1)
    cv2.imshow('gun',img_2)
    """1 1 1 
       1 1 1
       1 1 1"""
    SE1 = [(1,1),(1, 0), (1, -1), (0, 1), (0, 0),(0,-1),(-1,1),(-1,0),(-1,-1)]
    img_d1 = np.uint8(Dilation(img_1,SE1))
    cv2.imshow('Dilation palm',img_d1)
    img_d2 = np.uint8(Dilation(img_2,SE1))
    cv2.imshow('Dilation gun',img_d2)
    img_e1 = np.uint8(Erosion(img_1,SE1))
    cv2.imshow('Erosion palm',img_e1)
    img_e2 = np.uint8(Erosion(img_2,SE1))
    cv2.imshow('Erosion gun',img_e2)
    img_o1 = np.uint8(Opening(img_1,SE1))
    cv2.imshow('Opening palm',img_o1)
    img_o2 = np.uint8(Opening(img_2,SE1))
    cv2.imshow('Opening gun',img_o2)
    img_c1 = np.uint8(Closing(img_1,SE1))
    cv2.imshow('Closing palm',img_c1)
    img_c2 = np.uint8(Closing(img_2,SE1))
    cv2.imshow('Closing gun',img_c2)
    img_b1 = np.uint8(Boundary(img_c1))
    cv2.imshow('Boundary palm (reference closing palm)',img_b1)
    img_b2 = np.uint8(Boundary(img_c2))
    cv2.imshow('Boundary gun (reference closing guns)',img_b2)

    """  1 1 1 
       1 1 1 1 1
       1 1 1 1 1
       1 1 1 1 1
         1 1 1"""
    SE2 = [(2,1), (2, 0), (2, -1),(1,2),(1,1),(1, 0), (1, -1),(1,-2) , (0,2),(0, 1), (0, 0),(0,-1),(0,-2),(-1,2),(-1,1),(-1,0),(-1,-1),(-1,-2),(-2,1),(-2,0),(-2,-1)]

    img_d1 = np.uint8(Dilation(img_1,SE2))
    cv2.imshow('Dilation palm-star',img_d1)
    img_d2 = np.uint8(Dilation(img_2,SE2))
    cv2.imshow('Dilation gun-star',img_d2)
    img_e1 = np.uint8(Erosion(img_1,SE2))
    cv2.imshow('Erosion palm-star',img_e1)
    img_e2 = np.uint8(Erosion(img_2,SE2))
    cv2.imshow('Erosion gun-star',img_e2)
    img_o1 = np.uint8(Opening(img_1,SE2))
    cv2.imshow('Opening palm-star',img_o1)
    img_o2 = np.uint8(Opening(img_2,SE2))
    cv2.imshow('Opening gun-star',img_o2)
    img_c1 = np.uint8(Closing(img_1,SE2))
    cv2.imshow('Closing palm-star',img_c1)
    img_c2 = np.uint8(Closing(img_2,SE2))
    cv2.imshow('Closing gun-star',img_c2)
    img_b1 = np.uint8(Boundary(img_c1))
    cv2.imshow('Boundary palm (reference closing palm)-star',img_b1)
    img_b2 = np.uint8(Boundary(img_c2))
    cv2.imshow('Boundary gun (reference closing guns)-star',img_b2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
