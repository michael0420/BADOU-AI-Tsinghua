import cv2 as cv
import numpy as np
import math

def bilinear_inter(img):
	src_h,src_w = img.shape[:2]
	dst_img = np.zeros((1024,1024,3), np.uint8)
	dst_h,dst_w = dst_img.shape[:2]
	scale_h = src_h / dst_h
	scale_w = src_w / dst_w
	
	for c in range(3):
		for xx in range(dst_h):
			for yy in range(dst_w):
				x = (xx+0.5)*scale_h - 0.5
				y = (yy+0.5)*scale_w - 0.5
				x1 = math.floor(x)
				x2 = min(math.ceil(x),src_h-1)
				y1 = math.floor(y)
				y2 = min(math.ceil(y),src_w-1)
				Q11 = img[x1,y1,c]
				Q12 = img[x1,y2,c]
				Q21 = img[x2,y1,c]
				Q22 = img[x2,y2,c]
				P = (x2-x)*(y2-y)*Q11 + (x2-x)*(y-y1)*Q12 + (x-x1)*(y2-y)*Q21 + (x-x1)*(y-y1)*Q22
				dst_img[xx,yy,c]= int(P)
	return dst_img


if __name__ == '__main__':
	src_img = cv.imread('lenna.png')
	dst_img = bilinear_inter(src_img)
	cv.imshow('src image', src_img)
	cv.imshow('bilinear image', dst_img)
	cv.waitKey()
	

