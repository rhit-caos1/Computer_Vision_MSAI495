import numpy as np
import cv2

# img_1 = cv2.imread("test.bmp",cv2.IMREAD_GRAYSCALE)

# # cv2.imshow('img1',img_1)

# im = np.flipud(img_1)
# # print(im)
# print(np.shape(im))
# print(im[53][68])

# record = np.zeros(np.shape(im))

# L = 0

# for i in range(np.shape(im)[0]):
#     for j in range(np.shape(im)[1]):

#         if im[i][j] == 255:
#             # find up and left pixel
#             if i == 0:
#                 upper = 0
#             else:
#                 upper = record[i-1][j]
#             if j == 0:
#                 left = 0
#             else:
#                 left = record[i][j-1]
            

#             #decision making
#             if left == upper and left != 0 and upper != 0:
#                 record[i][j] = upper
#             elif left != upper and (left == 0 or upper == 0):
#                 record[i][j] = max(upper,left)
#             elif left != upper and left > 0 and upper > 0:
#                 record[i][j] = min(upper,left)
#                 # add set function
#             else:
#                 record[i][j] = L + 50

# print(record[53][68])


# cv2.imshow('img1',np.uint8(record))

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# def setlist(set_list, num1, num2):
#     # Check if the list is empty or none of the ints exist in the sets in the list
#     if not set_list or all(num not in s for s in set_list for num in (num1, num2)):
#         # Create a new set with the two ints
#         new_set = set((num1, num2))
#         # Append the new set to the list
#         set_list.append(new_set)
#     else:
#         # Iterate over the sets in the list
#         for s in set_list:
#             # If one of the ints is already in the set, add the other int
#             if num1 in s:
#                 s.add(num2)
#                 break
#             elif num2 in s:
#                 s.add(num1)
#                 break

def setlist(set_list, a, b):
    #check if num1 and num2 dose not exist in the list
    flag = False
    if len(set_list) != 0:
        for s in set_list:
            if a in s or b in s:
                flag = True
                if a in s:
                    s.add(b)
                    break
                elif b in s:
                    s.add(a)
                    break
        if flag == False:
            new_set = set((a,b))
            set_list.append(new_set)

    elif len(set_list) == 0 or flag == False:
        new_set = set((a,b))
        set_list.append(new_set)



def CCL(img):
    im = np.array(img)
    # print(im)
    print(np.shape(im))
    print(im[53][68])

    record = np.zeros(np.shape(im))

    L = 0
    set_list = []

    for i in range(np.shape(im)[0]):
        for j in range(np.shape(im)[1]):

            if im[i][j] == 255:
                # find up and left pixel
                if i == 0:
                    upper = 0
                else:
                    upper = record[i-1][j]
                if j == 0:
                    left = 0
                else:
                    left = record[i][j-1]
                

                #decision making
                if left == upper and left != 0 and upper != 0:
                    record[i][j] = upper
                elif left != upper and (left == 0 or upper == 0):
                    record[i][j] = max(upper,left)
                elif left != upper and left > 0 and upper > 0:
                    record[i][j] = min(upper,left)
                    # add set function
                    print(record[i][j])
                    setlist(set_list, upper, left)
                else:
                    L = L+1
                    record[i][j] = L


    print(set_list)
    
    for i in range(np.shape(record)[0]):
        for j in range(np.shape(record)[1]):
            for k in range(len(set_list)):
                if record[i][j] in set_list[k]:
                    record[i][j] = k+1

    return record*40

def size_filter(img,size):
    unique_id = np.unique(img)
    clear_id = []
    for i in unique_id:
        if i != 0 and np.count_nonzero(img == i) < size:
            clear_id.append(i)
    
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            for k in clear_id:
                if img [i][j] == k:
                    img [i][j] = 0

def main():
    img_1 = cv2.imread("test.bmp",cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread("face.bmp",cv2.IMREAD_GRAYSCALE)
    img_3 = cv2.imread("gun.bmp",cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img0',img_1)
    img_ccl1 = np.uint8(CCL(img_1))
    cv2.imshow('img1',img_ccl1)
    img_ccl2 = np.uint8(CCL(img_2))
    cv2.imshow('img2',img_ccl2)
    img_ccl3 = np.uint8(CCL(img_3))
    cv2.imshow('img3',img_ccl3)
    size_filter(img_ccl3,1000)
    cv2.imshow('img4',img_ccl3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

