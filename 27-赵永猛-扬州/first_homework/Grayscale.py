import numpy as np
import cv2


def BGR2GRAY(img):
    B = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    R = img[:, :, 2].copy()
    out = R * 0.299 + G * 0.587 + B * 0.114
    out = out.astype(np.uint8)
    return out


Src_image = cv2.imread("11-1.png")
Dst_image = BGR2GRAY(Src_image)
print(Dst_image)

# Show and Save
cv2.imshow('result1', Src_image)
cv2.imshow('result2', Dst_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
