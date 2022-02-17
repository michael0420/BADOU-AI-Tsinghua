import cv2 as cv
import numpy as np


def nearest_image(img):
	src_h, src_w, src_c = img.shape
	dst_img = np.zeros((700,700,src_c), np.uint8)
	dst_h, dst_w, dst_c = dst_img.shape
	scale_h = src_h/dst_h
	scale_w = src_w/dst_w
	
	for c in range(dst_c):
		for i in range(dst_h):
			for j in range(dst_w):
				x = int(i*scale_h)
				y = int(j*scale_w)
				dst_img[i,j,c] = img[x,y,c]
	
	return dst_img
			

if __name__ == '__main__':
	src_img = cv.imread("lenna.png")
	dst_img = nearest_image(src_img)
	cv.imshow('src_image', src_img)
	cv.imshow('nearest_image', dst_img)
	cv.waitKey()
	
	