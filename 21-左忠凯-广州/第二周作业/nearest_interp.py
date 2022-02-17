
import cv2
import numpy as np

def img_nearest_zoom(dest_h, dtst_w, img):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    img_zoom = np.zeros((dest_h, dtst_w, channels), np.uint8) # 空的图形数组

    height_scal = dest_h / height
    width_scal = dtst_w / width

    for i in range(dest_h):
        for j in range(dtst_w):

            s_x = int(i/height_scal)
            s_y = int(j/width_scal)

            img_zoom[i, j] = img[s_x, s_y]

    return img_zoom


img = cv2.imread("lenna.png")
new_img = img_nearest_zoom(800, 800, img)
print(new_img)
print(new_img.shape)

cv2.imshow("source img", img)
cv2.imshow("zoom img", new_img)
cv2.waitKey(0)

