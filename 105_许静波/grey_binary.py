import cv2
import numpy as np

def grey_binary(image_path):
    image = cv2.imread(image_path)
    h,w = image.shape[:2]
    image_grey = np.zeros([h,w],image.dtype)
    image_binary = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            p = image[i,j]
            image_grey[i,j] = int(0.11*p[0]+0.59*p[1]+0.3*p[2])
            if (image_grey[i][j] > 100):
                image_binary[i][j] = 1
            else:
                image_binary[i][j] = 0
    cv2.imshow("image", image)
    cv2.imshow("image grey", image_grey)
    cv2.imshow("image binary", image_binary)
    cv2.waitKey(0)
    print("image grey", image_grey)
    print("image binary", image_binary)



if __name__ == '__main__':
    image_path = 'lenna.png'
    grey_binary(image_path)

