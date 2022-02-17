import numpy as np
import cv2


def binarization(image):
    b = image[:, :, 0].copy()
    g = image[:, :, 1].copy()
    r = image[:, :, 2].copy()
    out = r*0.299 + g*0.587 + b*0.114
    out = out.astype(np.uint8)
    out[out > 128] = 255
    out[out < 128] = 0
    return out


image = cv2.imread("11-1.png")
out_ = binarization(image)
print(out_)

# Show and Save
cv2.imshow('result', out_)
cv2.waitKey(0)
cv2.destroyAllWindows()
