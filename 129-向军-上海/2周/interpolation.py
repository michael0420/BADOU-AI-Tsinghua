import cv2
import numpy as np
import config

# nearest interpolation
# use height and width scales to zoom
readImage = cv2.imread(config.lena_path)
height,width,channel = readImage.shape
zoomArr = np.zeros((800,800,channel),np.uint8)
print(readImage)
heightScale = 800 / height
widthScale = 800 / width
for j in range(800):
    for k in range(800):
        oriX = int(j / heightScale)
        oriY = int(k / widthScale)
        zoomArr[j,k] = readImage[oriX,oriY]
cv2.imshow("zoom_nearest",zoomArr)
cv2.waitKey(2000)
cv2.imwrite(config.test_out+"zoom_nearest.jpg",zoomArr)

# bilinear interpolation
def bilinearInterpolation(image, outShape):
    oriHeight,oriWidtd,channel = image.shape
    outHeight,outWidth = outShape[0],outShape[1]
    if oriHeight == outHeight and oriWidtd == outWidth:
        return  image.copy()
    outImage = np.zeros((outHeight,outWidth,channel),dtype = np.uint8)
    scaleX,scaleY = float(oriHeight) / outHeight, float(oriWidtd) / outWidth
    for c in range(channel):
        for i in range(outHeight):
            for j in range(outWidth):
                srcX = (j + 0.5) * scaleX - 0.5
                srcY = (i + 0.5) * scaleY - 0.5
                srcX0 = int(np.floor(srcX))
                srcY0 = int(np.floor(srcY))
                srcX1 = min(srcX0 + 1,oriWidtd - 1)
                srcY1 = min(srcY0 + 1,oriHeight - 1)
                temp0 = (srcX1 - srcX) * image[srcY0,srcX0,c] + (srcX - srcX0) * image[srcY0,srcX1,c]
                temp1 = (srcX1 - srcX) * image[srcY1,srcX0,c] + (srcX - srcX0) * image[srcY1,srcX1,c]
                outImage[i,j,c] = int((srcY1 - srcY) * temp0 + (srcY - srcY0) * temp1)
    return outImage
biOutImage = bilinearInterpolation(readImage,[800,800])
cv2.imshow("zoom_bilinear",zoomArr)
cv2.waitKey(2000)
cv2.imwrite(config.test_out+"zoom_bilinear.jpg",biOutImage)